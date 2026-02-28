"""
Mandelbrot Set Generator - naive approach
Author : Marko Šelendić
Course : Numerical Scientific Computing 2026
"""
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import time

from line_profiler import profile

from util import mandelbrot_time_test

matplotlib.use('TkAgg')


@profile
def mandelbrot_point(c, max_iter=100):
    """
    Calculates the number of iterations for a point c to escape the Mandelbrot set.

    Parameters
    ----------
    c : complex
        Complex point to evaluate
    max_iter : int
        Maximum number of iterations (default is 100)

    Returns
    -------
    int
        Number of iterations before escape, or max_iterations if it does not escape
    """
    z_0 = 0 + 0j
    z_n = z_0
    for n in range(max_iter):
        z_n = z_n ** 2 + c
        if abs(z_n) > 2:
            return n
    return max_iter


@profile
def compute_mandelbrot(C: np.ndarray, threshold=2, max_iterations=100, dtype=np.int32) -> np.ndarray:
    """
    Generates the Mandelbrot set (naively).

    Parameters
    ----------
    C: np.ndarray
        Array of complex numbers representing the points to evaluate
    threshold : float
        Escape threshold (default is 2)
    max_iterations : int
        Maximum number of iterations (default is 100)
    dtype : type
        Data type for the output array (default is np.int32)

    Returns
    -------
    np.ndarray
        2D array representing the Mandelbrot set
    """

    mandelbrot_set = np.zeros(C.shape, dtype=dtype)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            mandelbrot_set[i, j] = mandelbrot_point(C[i, j], max_iterations)

    return mandelbrot_set


@profile
def generate_complex_grid(image_size: int, dtype: int = np.complex128) -> np.ndarray:
    """
    Generate the complex grid for the Mandelbrot set.

    Parameters
    ----------
    image_size
        Size of the output image (default is 256)
    dtype : type
        Data type for the complex grid (default is np.complex128)

    Returns
    -------
        The complex grid for the Mandelbrot set as a 2D array of complex numbers.

    """
    X = np.linspace(-2, 1, image_size)
    Y = np.linspace(-1.5, 1.5, image_size)
    C = np.array([[complex(x, y) for x in X] for y in Y], dtype=dtype)

    return C


def main(image_size=4096):
    """Generates and displays the Mandelbrot set."""
    C = generate_complex_grid(image_size)
    start_time = time.perf_counter()
    mandelbrot_set = compute_mandelbrot(C)
    end_time = time.perf_counter()
    print(f"Mandelbrot set generated in {end_time - start_time:.2f} seconds.")

    plt.imshow(mandelbrot_set, extent=(-2, 1, -1.5, 1.5), cmap='viridis')
    plt.colorbar()
    plt.title("Mandelbrot Set")
    plt.show()


if __name__ == "__main__":
    # main(image_size=1024)

    results, medians, means, stddevs, image_sizes = mandelbrot_time_test(
        func_gen=generate_complex_grid,
        func_calc=compute_mandelbrot,
        top_size_log_2=4,
        n_runs_per_size=3
    )

    # Image size: 256x256, Time taken: 0.15 seconds
    # Image size: 512x512, Time taken: 0.60 seconds
    # Image size: 1024x1024, Time taken: 2.5 seconds
    # Image size: 2048x2048, Time taken: 9.5 seconds
    # Image size: 4096x4096, Time taken: 37 seconds
    # Image size: 8192x8192, Time taken: 150 seconds
    # (x4 per step?)
