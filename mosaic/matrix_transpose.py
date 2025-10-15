import os

from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
from jax.experimental.mosaic.gpu import profiler


@partial(jax.jit, static_argnames=['block_size'])
def pallas_transpose(x: jax.Array, block_size=128):
    m, k = x.shape
    @partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct((m, k), x.dtype),
        in_specs=[
            pl.BlockSpec((block_size, block_size), lambda i, j: (i, j)),
        ],
        out_specs=pl.BlockSpec((block_size, block_size), lambda i, j: (j, i)),
        grid=(m // block_size, k // block_size),
        compiler_params=plgpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")),
    )
    def transpose_kernel(i_ref, o_ref):
        o_ref[...] = jnp.transpose(i_ref[...], (1, 0))

    return transpose_kernel(x)


if __name__ == "__main__":
    size = 1024 * 8
    m, k = size, size
    data_size_gb = m * k * 2 * 2 / 1e9  # bfloat16
    x = jax.random.normal(jax.random.key(42), (m, k), dtype=jnp.bfloat16).block_until_ready()

    block_size = 128

    jax_transpose = jax.jit(jnp.transpose)

    repeat = 100

    # Warmup
    for _ in range(repeat):
        jax_transpose(x).block_until_ready()
        pallas_transpose(x, block_size).block_until_ready()

    out, runtimes_ms = profiler.measure(
        partial(pallas_transpose, block_size=block_size), iterations=repeat,
    )(x)
    runtime = np.median(runtimes_ms) / 1000
    bandwidth = data_size_gb / runtime
    print(f"pallas transpose {(m, k)=} {runtime=:.3f}s {bandwidth=:.2f} GB/s")

    out, runtimes_ms = profiler.measure(
        jax_transpose, iterations=repeat,
    )(x)
    runtime = np.median(runtimes_ms) / 1000
    bandwidth = data_size_gb / runtime
    print(f"jax transpose {(m, k)=} {runtime=:.3f}s {bandwidth=:.2f} GB/s")
