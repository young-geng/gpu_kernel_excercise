import os

from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
from jax.experimental.mosaic.gpu import profiler


@partial(jax.jit, static_argnames=['block_size'])
def pallas_matrix_add(
    x: jax.Array,
    y: jax.Array,
    block_size=(128, 128),
):

    smem_transforms = (
        plgpu.TilingTransform((8, 128 // 2)),
        plgpu.SwizzleTransform(128)
    )
    @partial(
        plgpu.kernel,
        out_shape=jax.ShapeDtypeStruct((x.shape[0], x.shape[1]), x.dtype),
        scratch_shapes=dict(
            smem_a_ref=plgpu.SMEM((block_size[0], block_size[1]), x.dtype, transforms=smem_transforms),
            smem_b_ref=plgpu.SMEM((block_size[0], block_size[1]), y.dtype, transforms=smem_transforms),
            barriers=plgpu.Barrier(num_barriers=2),
        ),
        grid=(x.shape[0] // block_size[0], x.shape[1] // block_size[1]),
        grid_names=("m", "n"),
    )
    def kernel(refs, o_ref, smem_a_ref, smem_b_ref, barriers):
        a_ref, b_ref = refs
        m_index = jax.lax.axis_index("m")
        n_index = jax.lax.axis_index("n")

        a_tile = a_ref.at[
            pl.ds(m_index * block_size[0], block_size[0]),
            pl.ds(n_index * block_size[1], block_size[1])
        ]

        b_tile = b_ref.at[
            pl.ds(m_index * block_size[0], block_size[0]),
            pl.ds(n_index * block_size[1], block_size[1])
        ]

        plgpu.copy_gmem_to_smem(a_tile, smem_a_ref, barriers.at[0])
        plgpu.copy_gmem_to_smem(b_tile, smem_b_ref, barriers.at[1])
        plgpu.barrier_wait(barriers.at[0])
        plgpu.barrier_wait(barriers.at[1])

        smem_a_ref[...] += smem_b_ref[...]

        o_tile = o_ref.at[
            pl.ds(m_index * block_size[0], block_size[0]),
            pl.ds(n_index * block_size[1], block_size[1])
        ]
        plgpu.copy_smem_to_gmem(smem_a_ref, o_tile)
        plgpu.wait_smem_to_gmem(0)

    return kernel((x, y))


if __name__ == "__main__":
    m, n = 8192, 8192
    matrix_add_data_size = m * n * 3 * 2 / 1e9  # bfloat16
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    x = jax.random.normal(k1, (m, n), dtype=jnp.bfloat16).block_until_ready()
    y = jax.random.normal(k2, (m, n), dtype=jnp.bfloat16).block_until_ready()

    block_size = (128, 128)

    jax_matrix_add = jax.jit(jnp.add)

    repeat = 100

    # Warmup
    for _ in range(repeat):
        jax_matrix_add(x, y).block_until_ready()
        pallas_matrix_add(x, y, block_size=block_size).block_until_ready()

    out, runtimes_ms = profiler.measure(
        partial(pallas_matrix_add, block_size=block_size), iterations=repeat,
    )(x, y)

    runtime = np.median(runtimes_ms) / 1000
    performance = matrix_add_data_size / runtime
    print(f"pallas matrix add {(m, n)=} {runtime=:.3f}s {performance=:.2f} GB/s")

    out, runtimes_ms = profiler.measure(
        jax_matrix_add, iterations=repeat,
    )(x, y)
    runtime = np.median(runtimes_ms) / 1000
    performance = matrix_add_data_size / runtime
    print(f"jax matrix add {(m, n)=} {runtime=:.3f}s {performance=:.2f} GB/s")
