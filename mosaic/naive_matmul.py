import os

from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
from jax.experimental.mosaic.gpu import profiler


def matmul_kernel(a_ref, b_ref, o_ref):
    @pl.when(pl.program_id(2) == 0)
    def _():
        o_ref[...] = jnp.zeros_like(o_ref)

    o_ref[...] += jnp.dot(a_ref[...], b_ref[...])


@partial(jax.jit, static_argnames=['block_size'])
def pallas_matmul(
    x: jax.Array,
    y: jax.Array,
    block_size=(64, 256, 32),
):
    m, k = x.shape
    _, n = y.shape
    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        in_specs=[
            pl.BlockSpec((block_size[0], block_size[1]), lambda i, j, k: (i, k)),
            pl.BlockSpec((block_size[1], block_size[2]), lambda i, j, k: (k, j))
        ],
        out_specs=pl.BlockSpec((block_size[0], block_size[2]), lambda i, j, k: (i, j)),
        grid=(m // block_size[0], n // block_size[2], k // block_size[1]),
        compiler_params=plgpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")),
    )(x, y)


if __name__ == "__main__":
    m, k, n = 4096, 8192, 4096
    matmul_tflop = 2 * m * n * k / 1e12
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    x = jax.random.normal(k1, (m, k), dtype=jnp.bfloat16).block_until_ready()
    y = jax.random.normal(k2, (k, n), dtype=jnp.bfloat16).block_until_ready()

    # block_size = (64, 256, 16)
    block_size = (128, 128, 128)

    jax_matmul = jax.jit(jnp.dot)

    repeat = 100

    # Warmup
    for _ in range(repeat):
        jax_matmul(x, y).block_until_ready()
        pallas_matmul(x, y, block_size=block_size).block_until_ready()

    out, runtimes_ms = profiler.measure(
        partial(pallas_matmul, block_size=block_size), iterations=repeat,
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

