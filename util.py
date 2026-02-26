import statistics, time

from matplotlib import pyplot as plt

from memory_profiler import profile

@profile
def benchmark(func_gen, func_calc, image_size: int, n_runs=3, **kwargs):
    """
    Time func, return median of n_runs.

    Parameters
    ----------
    func_gen : function
        Function to generate the complex grid for Mandelbrot set
    func_calc : function
        Function to perform the computation of the Mandelbrot set
    image_size : int
        Resolution of the Mandelbrot set
    n_runs : int
        Number of runs to perform for timing (default is 3)
    kwargs : dict
        Additional keyword arguments to pass to func_calc (e.g. threshold, max_iterations)
    """
    times = []
    result = None
    for _ in range(n_runs):
        C = func_gen(image_size)
        t0 = time.perf_counter()
        result = func_calc(C, **kwargs)
        times.append(time.perf_counter() - t0)
    median_t = statistics.median(times)
    print(f"Median: {median_t:.4f} s"
          f"(min = {min(times):.4f}, max = {max(times):.4f})")
    return median_t, result


def mandelbrot_time_test(func_gen, func_calc,
                         start_size_log_2: int = 0, top_size_log_2: int = 5,
                         n_runs_per_size: int = 3,
                         show_plots: bool = True,
                         **kwargs):
    """
    Runs the Mandelbrot set generator for different image resolutions,
    measures the time taken for each and plots the results.

    Parameters
    ----------
    func_gen : function
        Function to generate the complex grid for Mandelbrot set
    func_calc : function
        Function to perform the computation of the Mandelbrot set
    start_size_log_2 : int
        The logarithm base 2 of the coefficient for the smallest image size to test (default is 0 which corresponds to 256 = 256 * 2^0)
    top_size_log_2 : int
        The logarithm base 2 of the coefficient for the largest image size to test (default is 5 which corresponds to 8192 = 256 * 2^5)
    n_runs_per_size : int
        Number of runs to perform for timing each image size (default is 3)
    kwargs : dict
        Additional keyword arguments to pass to func_calc (e.g. threshold, max_iterations)
    """
    image_sizes = [256 * 2 ** i for i in range(start_size_log_2, top_size_log_2+1)]  # 256, 512, 1024, 2048, 4096, 8192 if top_size_log_2 = 5
    times = []
    for size in image_sizes:
        print(f"\nImage size {size}x{size}:")
        median, result = benchmark(func_gen, func_calc, image_size=size, n_runs=n_runs_per_size, **kwargs)
        times.append(median)
        if show_plots:
            plt.plot(image_sizes, times, marker='o')
            plt.show()
    return times, image_sizes
