import os

from functools import partial
import timeit
import numpy as np
import jax
import jax.numpy as jnp

import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu


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
    size = 1024 * 16
    m, k, n = size, size, size
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    x = jax.random.normal(k1, (m, k), dtype=jnp.bfloat16).block_until_ready()
    y = jax.random.normal(k2, (k, n), dtype=jnp.bfloat16).block_until_ready()

    # block_size = (64, 256, 16)
    block_size = (128, 128, 128)

    def run_matmul_pallas():
        pallas_matmul(x, y, block_size=block_size).block_until_ready()

    def run_matmul_jax():
        jax.jit(jnp.dot)(x, y).block_until_ready()

    repeat = 10

    # Warmup
    for _ in range(repeat):
        run_matmul_pallas()
        run_matmul_jax()

    time = timeit.timeit(
        "run_matmul_pallas()",
        number=repeat,
        globals=globals(),
    ) / repeat
    performance = m * n * k / time / 1e12
    print(f"pallas matmul {size=} {time=:.3f}s {performance=:.2f} TFLOPS")


    time = timeit.timeit(
        "run_matmul_jax()",
        number=repeat,
        globals=globals(),
    ) / repeat
    performance = m * n * k / time / 1e12
    print(f"jax matmul {size=} {time=:.3f}s {performance=:.2f} TFLOPS")
