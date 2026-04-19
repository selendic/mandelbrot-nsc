"""
Mandelbrot Set Generator - numba-jit approach
Author : Marko Šelendić
Course : Numerical Scientific Computing 2026
"""

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from numba import njit, prange

from naive import mandelbrot_naive
from numpy_simd import mandelbrot_numpy
from util import mandelbrot_time_test

matplotlib.use('TkAgg')


def generate_complex_grid(image_size: int, dtype=np.complex128) -> np.ndarray:
    """
    Generate the complex grid for the Mandelbrot set.

    Parameters
    ----------
    image_size
        Size of the output image (default is 256)
    dtype
        Data type for the complex grid (default is np.complex128)

    Returns
    -------
    np.ndarray
        The complex grid for the Mandelbrot set as a 2D array of complex numbers.

    """

    if dtype == np.complex128:
        dtype = np.float64
    elif dtype == np.complex64:
        dtype = np.float32
    else:
        raise ValueError("Unsupported dtype for complex grid. Use np.complex128 or np.complex64.")

    xs = np.linspace(-2, 1, image_size, dtype=dtype)
    ys = np.linspace(-1.5, 1.5, image_size, dtype=dtype)
    X, Y = np.meshgrid(xs, ys)
    C = X + 1j * Y

    return C


@njit
def mandelbrot_point_numba(c, t: float = 4.0, max_iter=100):
    """
    Calculates the number of iterations for a point c to escape the Mandelbrot set (numba-njit optimized).

    Parameters
    ----------
    c : complex
        Complex point to evaluate
    t : float
        Escape threshold squared (to avoid computing square root; default is 4.0 which corresponds to a threshold of 2)
    max_iter : int
        Maximum number of iterations (default is 100)

    Returns
    -------
    int
        Number of iterations before escape, or max_iterations if it does not escape
    """
    z = 0j
    for n in range(max_iter):
        z = z * z + c
        if z.real * z.real + z.imag * z.imag > t:
            return n
    return max_iter


def compute_mandelbrot_hybrid(C: np.ndarray, threshold=2, max_iter=100, dtype=np.int32) -> np.ndarray:
    """
    Generates the Mandelbrot set (naively).

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
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            mandelbrot_set[i, j] = mandelbrot_point_numba(C[i, j], t, max_iter)

    return mandelbrot_set


@njit
def mandelbrot_naive_full_numba(C: np.ndarray, threshold=2, max_iter=100, dtype=np.int32) -> np.ndarray:
    """
    Generates the Mandelbrot set (fully numba-njit optimized).

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
    for i in range(C.shape[0]):
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


def main(
        image_size_start_log_2: int = 0,
        image_size_top_log_2: int = 4,
        dtype_c: type = np.complex128,
        dtype_out: type = np.int32,
        runs_per_size: int = 10,
        warmup_runs: int = 2
):
    """Benchmark naive, NumPy, and Numba Mandelbrot implementations across sizes."""
    results_naive, medians_naive, means_naive, stddevs_naive, image_sizes = mandelbrot_time_test(
        func_gen=mandelbrot_naive.generate_complex_grid,
        func_calc=mandelbrot_naive.compute_mandelbrot,
        start_size_log_2=image_size_start_log_2,
        top_size_log_2=image_size_top_log_2,
        n_runs_per_size=runs_per_size,
        warmup_runs=0,
        show_plots=False,
        dtype_c=dtype_c,
        dtype_out=dtype_out
    )

    results_numpy, medians_numpy, means_numpy, stddevs_numpy, _ = mandelbrot_time_test(
        func_gen=mandelbrot_numpy.generate_complex_grid,
        func_calc=mandelbrot_numpy.compute_mandelbrot,
        start_size_log_2=image_size_start_log_2,
        top_size_log_2=image_size_top_log_2,
        n_runs_per_size=runs_per_size,
        warmup_runs=0,
        show_plots=False,
        dtype_c=dtype_c,
        dtype_out=dtype_out
    )

    results_numba_hybrid, medians_numba_hybrid, means_numba_hybrid, stddevs_numba_hybrid, _ = mandelbrot_time_test(
        func_gen=generate_complex_grid,
        func_calc=compute_mandelbrot_hybrid,
        start_size_log_2=image_size_start_log_2,
        top_size_log_2=image_size_top_log_2,
        n_runs_per_size=runs_per_size,
        warmup_runs=warmup_runs,
        show_plots=False,
        dtype_c=dtype_c,
        dtype_out=dtype_out
    )

    results_numba_full, medians_numba_full, means_numba_full, stddevs_numba_full, _ = mandelbrot_time_test(
        func_gen=generate_complex_grid,
        func_calc=mandelbrot_naive_full_numba,
        start_size_log_2=image_size_start_log_2,
        top_size_log_2=image_size_top_log_2,
        n_runs_per_size=runs_per_size,
        warmup_runs=warmup_runs,
        show_plots=False,
        dtype_c=dtype_c,
        dtype_out=dtype_out
    )

    results_numba_full_parallel, medians_numba_full_parallel, means_numba_full_parallel, stddevs_numba_full_parallel, _ = mandelbrot_time_test(
        func_gen=generate_complex_grid,
        func_calc=mandelbrot_naive_full_numba_parallel,
        start_size_log_2=image_size_start_log_2,
        top_size_log_2=image_size_top_log_2,
        n_runs_per_size=runs_per_size,
        warmup_runs=warmup_runs,
        show_plots=False,
        dtype_c=dtype_c,
        dtype_out=dtype_out
    )

    for i in range(len(image_sizes)):
        print(f"\nImage size {image_sizes[i]}x{image_sizes[i]}:")
        all_close = np.allclose(results_numpy[i], results_naive[i]) and np.allclose(results_numba_hybrid[i],
                                                                                    results_naive[i]) and np.allclose(
            results_numba_full[i], results_naive[i]) and np.allclose(results_numba_full_parallel[i], results_naive[i])
        print(f"Results are close enough: {all_close}")

        print(f"Mandelbrot set generation times (avg ± std) [median speedup]:")
        print(f"  - Naive:               {means_naive[i]:.4f} ± {stddevs_naive[i]:.4f}s  [1.00x]")
        print(
            f"  - Numpy:               {means_numpy[i]:.4f} ± {stddevs_numpy[i]:.4f}s  [{medians_naive[i] / medians_numpy[i]:.2f}x]")
        print(
            f"  - Numba Hybrid:        {means_numba_hybrid[i]:.4f} ± {stddevs_numba_hybrid[i]:.4f}s  [{medians_naive[i] / medians_numba_hybrid[i]:.2f}x]")
        print(
            f"  - Numba Full:          {means_numba_full[i]:.4f} ± {stddevs_numba_full[i]:.4f}s  [{medians_naive[i] / medians_numba_full[i]:.2f}x]")
        print(
            f"  - Numba Full Parallel: {means_numba_full_parallel[i]:.4f} ± {stddevs_numba_full_parallel[i]:.4f}s  [{medians_naive[i] / medians_numba_full_parallel[i]:.2f}x]")

    # Plot time vs grid size for all implementations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Top-left plot - Normal scale
    ax1.plot(image_sizes[:len(medians_naive)], medians_naive, marker='o', label='Naive')
    ax1.plot(image_sizes[:len(medians_numpy)], medians_numpy, marker='o', label='Numpy')
    ax1.plot(image_sizes[:len(medians_numba_hybrid)], medians_numba_hybrid, marker='o', label='Numba Hybrid')
    ax1.plot(image_sizes[:len(medians_numba_full)], medians_numba_full, marker='o', label='Numba Full')
    ax1.plot(image_sizes[:len(medians_numba_full_parallel)], medians_numba_full_parallel, marker='o',
             label='Numba Full Parallel')
    ax1.set_xlabel('Image Size (pixels)')
    ax1.set_ylabel('Median Time (seconds)')
    ax1.set_title('Mandelbrot Set Generation Time - Normal Scale')
    ax1.grid(True, ls="--")
    ax1.legend()

    # Top-right plot - Log scale for both axes
    ax2.plot(image_sizes[:len(medians_naive)], medians_naive, marker='o', label='Naive')
    ax2.plot(image_sizes[:len(medians_numpy)], medians_numpy, marker='o', label='Numpy')
    ax2.plot(image_sizes[:len(medians_numba_hybrid)], medians_numba_hybrid, marker='o', label='Numba Hybrid')
    ax2.plot(image_sizes[:len(medians_numba_full)], medians_numba_full, marker='o', label='Numba Full')
    ax2.plot(image_sizes[:len(medians_numba_full_parallel)], medians_numba_full_parallel, marker='o',
             label='Numba Full Parallel')
    ax2.set_xlabel('Image Size (pixels)')
    ax2.set_ylabel('Median Time (seconds)')
    ax2.set_title('Mandelbrot Set Generation Time - Log Both Axes')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, which="both", ls="--")
    ax2.legend()

    # Bottom-left plot - Speedup relative to naive implementation (linear scale)
    speedup_numpy = [medians_naive[i] / medians_numpy[i] for i in range(len(medians_naive))]
    speedup_numba_hybrid = [medians_naive[i] / medians_numba_hybrid[i] for i in range(len(medians_naive))]
    speedup_numba_full = [medians_naive[i] / medians_numba_full[i] for i in range(len(medians_naive))]
    speedup_numba_full_parallel = [medians_naive[i] / medians_numba_full_parallel[i] for i in range(len(medians_naive))]

    ax3.plot(image_sizes[:len(speedup_numpy)], speedup_numpy, marker='o', label='Numpy')
    ax3.plot(image_sizes[:len(speedup_numba_hybrid)], speedup_numba_hybrid, marker='o', label='Numba Hybrid')
    ax3.plot(image_sizes[:len(speedup_numba_full)], speedup_numba_full, marker='o', label='Numba Full')
    ax3.plot(image_sizes[:len(speedup_numba_full_parallel)], speedup_numba_full_parallel, marker='o',
             label='Numba Full Parallel')
    ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Naive (baseline)')
    ax3.set_xlabel('Image Size (pixels)')
    ax3.set_ylabel('Speedup Factor (x)')
    ax3.set_title('Speedup Relative to Naive Implementation')
    ax3.grid(True, ls="--")
    ax3.legend()

    # Bottom-right plot - Speedup with log y-axis
    ax4.plot(image_sizes[:len(speedup_numpy)], speedup_numpy, marker='o', label='Numpy')
    ax4.plot(image_sizes[:len(speedup_numba_hybrid)], speedup_numba_hybrid, marker='o', label='Numba Hybrid')
    ax4.plot(image_sizes[:len(speedup_numba_full)], speedup_numba_full, marker='o', label='Numba Full')
    ax4.plot(image_sizes[:len(speedup_numba_full_parallel)], speedup_numba_full_parallel, marker='o',
             label='Numba Full Parallel')
    ax4.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Naive (baseline)')
    ax4.set_xlabel('Image Size (pixels)')
    ax4.set_ylabel('Speedup Factor (x)')
    ax4.set_title('Speedup Relative to Naive Implementation - Log Y-axis')
    ax4.set_yscale('log')
    ax4.grid(True, which="both", ls="--")
    ax4.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
