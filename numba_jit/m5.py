import numpy as np

from numba_jit.mandelbrot_numba_jit import generate_complex_grid, mandelbrot_naive_full_numba, \
    mandelbrot_naive_full_numba_parallel
from util import mandelbrot_time_test


def main1():
    """Benchmark the non-parallel full-Numba Mandelbrot implementation."""
    print("Testing 1024x1024 res full numba njit")
    dtype_c = np.complex128
    dtype_out = np.int32
    print(f"dtype_c={dtype_c.__name__} and dtype_out={dtype_out.__name__}:")
    mandelbrot_time_test(
        func_gen=generate_complex_grid,
        func_calc=mandelbrot_naive_full_numba,
        start_size_log_2=2,
        top_size_log_2=2,
        n_runs_per_size=10,
        warmup_runs=1,
        dtype_c=dtype_c,
        dtype_out=dtype_out,
        show_plots=False
    )


def main2():
    """Benchmark the parallel full-Numba Mandelbrot implementation."""
    print("Testing 1024x1024 res full numba njit parallel")
    dtype_c = np.complex128
    dtype_out = np.int32
    print(f"dtype_c={dtype_c.__name__} and dtype_out={dtype_out.__name__}:")
    mandelbrot_time_test(
        func_gen=generate_complex_grid,
        func_calc=mandelbrot_naive_full_numba_parallel,
        start_size_log_2=2,
        top_size_log_2=2,
        n_runs_per_size=10,
        warmup_runs=3,
        dtype_c=dtype_c,
        dtype_out=dtype_out,
        show_plots=False
    )


if __name__ == "__main__":
    # main1()
    main2()
