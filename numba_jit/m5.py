import numpy as np
from itertools import product
from matplotlib import pyplot as plt
from numba import njit, prange

from numba_jit.mandelbrot_numba_jit import generate_complex_grid, mandelbrot_naive_full_numba
from util import mandelbrot_time_test


def main1():
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


@njit(parallel=True)
def mandelbrot_naive_full_numba_parallel(C: np.ndarray, threshold=2, max_iter=100, dtype=np.int32) -> np.ndarray:
    """
    Generates the Mandelbrot set (fully numba-njit optimized AND parallelized).

    Parameters
    ----------
    C: np.ndarray
        Array of complex numbers representing the points to evaluate
    threshold : float
        Escape threshold (default is 2)
    max_iter : int
        Maximum number of iterations (default is 100)
    dtype : type
        Data type for the output array (default is np.int32)

    Returns
    -------
    np.ndarray
        2D array representing the Mandelbrot set
    """
    t = threshold * threshold
    mandelbrot_set = np.zeros(C.shape, dtype=dtype)
    for i in prange(C.shape[0]):
        for j in range(C.shape[1]):
            z = 0j
            for n in range(max_iter):
                z = z * z + C[i, j]
                if z.real * z.real + z.imag * z.imag > t:
                    mandelbrot_set[i, j] = n
                    break
            else:
                mandelbrot_set[i, j] = max_iter

    return mandelbrot_set


def main2():
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
