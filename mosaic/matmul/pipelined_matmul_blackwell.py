import os

from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
from jax.experimental.mosaic.gpu import profiler


@partial(jax.jit, static_argnames=['block_size', 'num_buffers'])
def pallas_matmul(
    x: jax.Array,
    y: jax.Array,
    block_size=(128, 128, 128),
    num_buffers=3,
):

    swizzle = plgpu.find_swizzle(block_size[1] * jnp.dtype(x.dtype).itemsize * 8)
    swizzle_elems = swizzle // jnp.dtype(x.dtype).itemsize
    smem_transforms = (
        plgpu.TilingTransform((8, swizzle_elems)),
        plgpu.SwizzleTransform(swizzle)
    )
    @partial(
        plgpu.kernel,
        out_shape=jax.ShapeDtypeStruct((x.shape[0], y.shape[1]), x.dtype),
        scratch_shapes=dict(
            smem_a_ref=plgpu.SMEM((num_buffers, block_size[0], block_size[1]), x.dtype, transforms=smem_transforms),
            smem_b_ref=plgpu.SMEM((num_buffers, block_size[1], block_size[2]), x.dtype, transforms=smem_transforms),
            tmem_acc_ref=plgpu.TMEM((block_size[0], block_size[2]), jnp.float32),
            smem_acc_ref=plgpu.SMEM((block_size[0], block_size[2]), x.dtype, transforms=smem_transforms),
            smem_a_barrier_ref=plgpu.Barrier(orders_tensor_core=False, num_barriers=num_buffers),
            smem_b_barrier_ref=plgpu.Barrier(orders_tensor_core=False, num_barriers=num_buffers),
            tc_barrier_ref=plgpu.Barrier(orders_tensor_core=True, num_barriers=num_buffers),
            tc_done_barrier_ref=plgpu.Barrier(orders_tensor_core=True, num_barriers=1),
        ),
        grid=(x.shape[0] // block_size[0], y.shape[1] // block_size[2]),
        grid_names=("m", "n"),
        num_threads=2,
        thread_name="pallas_thread",
    )
    def matmul_kernel(in_refs, o_ref, smem_a_ref, smem_b_ref, tmem_acc_ref, smem_acc_ref, smem_a_barrier_ref, smem_b_barrier_ref, tc_barrier_ref, tc_done_barrier_ref):
        a_ref, b_ref = in_refs
        m_index = jax.lax.axis_index("m")
        n_index = jax.lax.axis_index("n")
        thread_id = jax.lax.axis_index("pallas_thread")

        @pl.when(thread_id == 0)
        def load_data_thread():
            for k in range(x.shape[1] // block_size[1]):
                slot = k % num_buffers
                a_ref_tile = a_ref.at[
                    pl.ds(m_index * block_size[0], block_size[0]),
                    pl.ds(k * block_size[1], block_size[1])
                ]
                b_ref_tile = b_ref.at[
                    pl.ds(k * block_size[1], block_size[1]),
                    pl.ds(n_index * block_size[2], block_size[2])
                ]
                if k >= num_buffers:
                    # Wait for previous MMA at slot to finish using this buffer
                    plgpu.barrier_wait(tc_barrier_ref.at[slot])

                plgpu.copy_gmem_to_smem(
                    a_ref_tile,
                    smem_a_ref.at[slot, ...],
                    smem_a_barrier_ref.at[slot]
                )
                plgpu.copy_gmem_to_smem(
                    b_ref_tile,
                    smem_b_ref.at[slot, ...],
                    smem_b_barrier_ref.at[slot]
                )

        @pl.when(thread_id == 1)
        def mma_thread():
            for k in range(x.shape[1] // block_size[1]):
                slot = k % num_buffers
                # Wait for data to be ready at slot
                plgpu.barrier_wait(smem_a_barrier_ref.at[slot])
                plgpu.barrier_wait(smem_b_barrier_ref.at[slot])
                plgpu.tcgen05_mma(
                    tmem_acc_ref,
                    smem_a_ref.at[slot, ...],
                    smem_b_ref.at[slot, ...],
                    tc_barrier_ref.at[slot],
                    accumulate=k > 0
                )

            plgpu.tcgen05_commit_arrive(tc_done_barrier_ref)

        plgpu.barrier_wait(tc_done_barrier_ref)
        smem_acc_ref[...] = plgpu.async_load_tmem(tmem_acc_ref).astype(o_ref.dtype)
        plgpu.wait_load_tmem()
        plgpu.commit_smem()

        o_ref_tile = o_ref.at[
            pl.ds(m_index * block_size[0], block_size[0]),
            pl.ds(n_index * block_size[2], block_size[2])
        ]
        plgpu.copy_smem_to_gmem(smem_acc_ref, o_ref_tile)
        plgpu.wait_smem_to_gmem(0)

    return matmul_kernel((x, y))


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
