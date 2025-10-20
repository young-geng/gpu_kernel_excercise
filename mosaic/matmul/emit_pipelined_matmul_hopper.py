import os

from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
from jax.experimental.mosaic.gpu import profiler


@partial(jax.jit, static_argnames=['block_size', 'compute_wgs', 'max_concurrent_steps', 'memory_registers'])
def pallas_matmul(
    x: jax.Array,
    y: jax.Array,
    block_size=(128, 64, 128),
    compute_wgs=2,
    max_concurrent_steps=2,
    memory_registers=40,
):
    swizzle = plgpu.find_swizzle(block_size[1] * jnp.dtype(x.dtype).itemsize * 8)
    swizzle_elems = swizzle // jnp.dtype(x.dtype).itemsize
    transforms = (
        plgpu.TilingTransform((8, swizzle_elems)),
        plgpu.SwizzleTransform(swizzle)
    )
    BM, BK, BN = block_size
    assert x.shape[1] % BK == 0 and y.shape[1] % (BN * compute_wgs) == 0
    grid_m = x.shape[0] // BM
    grid_k = x.shape[1] // BK
    grid_n = y.shape[1] // (BN * compute_wgs)

    @partial(
        plgpu.kernel,
        out_shape=jax.ShapeDtypeStruct((x.shape[0], y.shape[1]), x.dtype),
        scratch_shapes=dict(
            o_smem=plgpu.SMEM((BM, BN * compute_wgs), x.dtype)
        ),
        grid=(grid_m, grid_n),
        grid_names=("m", "n"),
        num_threads=compute_wgs + 1,
        thread_name="wg",
    )
    def matmul_ws_kernel(a_gmem, b_gmem, o_gmem, o_smem):
        pid_m = pl.program_id(0)
        pid_n = pl.program_id(1)

        wg_idx = jax.lax.axis_index("wg")
        wg_slice = pl.ds(wg_idx * BN, BN)

        def compute_thread(pipeline):
            acc = plgpu.layout_cast(
                jnp.zeros((BM, BN), dtype=jnp.float32),
                plgpu.Layout.WGMMA,
            )
            final_acc = pipeline(acc)
            o_smem[:, wg_slice] = final_acc[...].astype(o_gmem.dtype)

        def kernel_body(_, a_smem, b_smem, carry):
            acc = carry
            b_wg = b_smem.at[:, wg_slice]

            def do_wgmma(acc_ref):
                plgpu.wgmma(acc_ref, a_smem, b_wg)
                # plgpu.wgmma_wait(1)

            acc = pl.run_state(do_wgmma)(plgpu.ACC.init(acc))
            return acc

        pipeline = plgpu.emit_pipeline_warp_specialized(
            kernel_body,
            in_specs=[
                plgpu.BlockSpec(
                    (BM, BK), lambda kk: (pid_m, kk), transforms=transforms
                ),
                plgpu.BlockSpec(
                    (BK, BN * compute_wgs), lambda kk: (kk, pid_n), transforms=transforms
                ),
            ],
            grid=(grid_k,),
            compute_context=compute_thread,
            max_concurrent_steps=max_concurrent_steps,
            num_compute_wgs=compute_wgs,
            memory_registers=memory_registers,
            wg_axis="wg",
            memory_thread_idx=compute_wgs,
        )

        pipeline(a_gmem, b_gmem)

        plgpu.commit_smem()
        m_slice = pl.ds(pid_m * BM, BM)
        n_slice = pl.ds(pid_n * BN * compute_wgs, BN * compute_wgs)
        plgpu.copy_smem_to_gmem(o_smem, o_gmem.at[m_slice, n_slice])
        plgpu.wait_smem_to_gmem(0)

    return matmul_ws_kernel(x, y)


if __name__ == "__main__":
    m, k, n = 4096, 8192, 4096
    matmul_tflop = 2 * m * n * k / 1e12
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    x = jax.random.normal(k1, (m, k), dtype=jnp.bfloat16).block_until_ready()
    y = jax.random.normal(k2, (k, n), dtype=jnp.bfloat16).block_until_ready()

    jax_matmul = jax.jit(jnp.dot)

    repeat = 100

    # Warmup
    for _ in range(repeat):
        jax_matmul(x, y).block_until_ready()
        pallas_matmul(x, y).block_until_ready()

    out, runtimes_ms = profiler.measure(
        partial(pallas_matmul), iterations=repeat,
    )(x, y)
    runtime = np.median(runtimes_ms) / 1000
    performance = matmul_tflop / runtime
    print(f"pallas matmul {(m, k, n)=} {runtime=:.3f}s {performance=:.2f} TFLOPS")

    out, runtimes_ms = profiler.measure(
        jax_matmul, iterations=repeat,
    )(x, y)
    runtime = np.median(runtimes_ms) / 1000
    performance = matmul_tflop / runtime
    print(f"jax matmul {(m, k, n)=} {runtime=:.3f}s {performance=:.2f} TFLOPS")
