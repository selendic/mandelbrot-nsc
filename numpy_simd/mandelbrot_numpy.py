"""
Mandelbrot Set Generator - numpy approach
Author : Marko Šelendić
Course : Numerical Scientific Computing 2026
"""
import cProfile, pstats
import time

import matplotlib

import numpy as np
from matplotlib import pyplot as plt

from naive import mandelbrot_naive as naive
from util import mandelbrot_time_test

matplotlib.use('TkAgg')

def compute_mandelbrot(C: np.ndarray, threshold: int = 2, max_iter=100) -> np.ndarray:
    """
    Calculates the number of iterations for each point in the array to escape the Mandelbrot set.

    Parameters
    ----------
    C: np.ndarray
        2D array of complex numbers representing the points to evaluate
    threshold : float
        Escape threshold (default is 2)
    max_iter : int
        Maximum number of iterations (default is 100)

    Returns
    -------
    int
        2D array of integers representing the number of iterations before escape, or max_iterations if one does not escape
    """

    z = np.zeros_like(C, dtype=np.complex128)
    output = np.zeros(C.shape, dtype=int)
    mask = np.ones(C.shape, dtype=bool)

    for n in range(max_iter):
        z[mask] = z[mask] ** 2 + C[mask]
        escaped = np.abs(z) > threshold
        output[escaped & mask] = n
        mask &= ~escaped

    output[mask] = max_iter
    return output


def generate_complex_grid(image_size: int) -> np.ndarray:
    """
    Generate the complex grid for the Mandelbrot set.

    Parameters
    ----------
    image_size : int
        Size of the output image (default is 256)

    Returns
    -------
        The complex grid for the Mandelbrot set as a 2D array of complex numbers.

    """
    xs = np.linspace(-2, 1, image_size)
    ys = np.linspace(-1.5, 1.5, image_size)
    X, Y = np.meshgrid(xs, ys)
    C = X + 1j * Y

    return C


def main(image_size=4096):
    """Generates and displays the Mandelbrot set."""
    C = generate_complex_grid(image_size)

    start_time_naive = time.perf_counter()
    mandelbrot_set_naive = naive.compute_mandelbrot(C, image_size=image_size)
    end_time_naive = time.perf_counter()

    start_time_np = time.perf_counter()
    mandelbrot_set_np = compute_mandelbrot(C)
    end_time_np = time.perf_counter()
    print(f"Mandelbrot set generated in:\n"
            f"- Naive: {end_time_naive - start_time_naive:.2f} seconds\n"
            f"- Numpy: {end_time_np - start_time_np:.2f} seconds")

    all_close = np.allclose(mandelbrot_set_naive, mandelbrot_set_np)
    print(f"Results are close enough: {all_close}")

    plt.imshow(mandelbrot_set_np, extent=(-2, 1, -1.5, 1.5), cmap='viridis')
    plt.colorbar()
    plt.title("Mandelbrot Set")
    plt.show()


if __name__ == "__main__":
    # main(image_size=4096)

    times, image_sizes = mandelbrot_time_test(
        func_gen=generate_complex_grid,
        func_calc=compute_mandelbrot,
        top_size_log_2=4,
        n_runs_per_size=3
    )
