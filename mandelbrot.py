"""
Mandelbrot Set Generator
Author : Marko Šelendić
Course : Numerical Scientific Computing 2026
"""
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import time


def f(c, max_iter=100):
    """
    Calculates the number of iterations for a point c to escape the Mandelbrot set.

    Parameters
    ----------
    c : complex
        Complex point to evaluate

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


def mandelbrot(threshold=2, max_iterations=100, image_size=256):
    """
    Generates the Mandelbrot set.

    Parameters
    ----------
    threshold : float
        Escape threshold (default is 2)
    max_iterations : int
        Maximum number of iterations (default is 100)
    image_size : int
        Size of the output image (default is 256)

    Returns
    -------
    np.ndarray
        2D array representing the Mandelbrot set
    """

    X = np.linspace(-2, 1, image_size)
    Y = np.linspace(-1.5, 1.5, image_size)
    C = np.array([[complex(x, y) for x in X] for y in Y])

    mandelbrot_set = np.zeros(C.shape, dtype=int)
    for i in range(image_size):
        for j in range(image_size):
            mandelbrot_set[i, j] = f(C[i, j], max_iterations)

    return mandelbrot_set


def mandelbrot_time_test():
    image_sizes = [256*2**i for i in range(5)]  # 256, 512, 1024, 2048, 4096
    times = []
    for size in image_sizes:
        start_time = time.time()
        mandelbrot(image_size=size)
        end_time = time.time()
        times.append(end_time - start_time)
        print(f"Image size: {size}x{size}, Time taken: {end_time - start_time:.2f} seconds")
    plt.plot(image_sizes, times, marker='o')
    plt.show()

def main():
    start_time = time.time()
    mandelbrot_set = mandelbrot(image_size=8192)
    end_time = time.time()
    print(f"Mandelbrot set generated in {end_time - start_time:.2f} seconds.")

    matplotlib.use('TkAgg')
    plt.imshow(mandelbrot_set, extent=(-2, 1, -1.5, 1.5), cmap='viridis')
    plt.colorbar()
    plt.title("Mandelbrot Set")
    plt.show()


if __name__ == "__main__":
    mandelbrot_time_test()
