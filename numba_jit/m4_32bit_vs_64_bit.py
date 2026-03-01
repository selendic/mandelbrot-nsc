import numpy as np
from itertools import product
from matplotlib import pyplot as plt

from numba_jit.mandelbrot_numba_jit import generate_complex_grid, mandelbrot_naive_full_numba
from util import mandelbrot_time_test


def main(show_plots: bool = False):
    print("Testing 1024x1024 res full numba")
    results = []
    for dtype_c, dtype_out in product([np.complex64, np.complex128], [np.int32]):
        print(f"\n{'-' * 60}")
        print(f"dtype_c={dtype_c.__name__} and dtype_out={dtype_out.__name__}:")
        result, _, _, _, _ = mandelbrot_time_test(
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
        results.append((dtype_c, dtype_out, result[0]))

    if not show_plots:
        # Plot the Mandelbrot sets for both data type combinations
        for dtype_c, dtype_out, mandelbrot_set in results:
            plt.imshow(mandelbrot_set, extent=(-2, 1, -1.5, 1.5), cmap='viridis')
            plt.title(f"Mandelbrot Set (dtype_c={dtype_c.__name__}, dtype_out={dtype_out.__name__})")
            plt.xlabel("Real Axis")
            plt.ylabel("Imaginary Axis")
            plt.colorbar(label='Iterations to Diverge')
            plt.show()


if __name__ == "__main__":
    main(show_plots=False)
