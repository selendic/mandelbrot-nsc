import statistics, time

from matplotlib import pyplot as plt


# from memory_profiler import profile

# @profile
def benchmark(func_gen, func_calc, image_size: int,
              n_runs: int = 3, warmup_runs: int = 0,
              dtype_c: type = None, dtype_out: type = None,
              **kwargs):
    """
    Time func, return median, mean, and stddev of n_runs.

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
    warmup_runs : int
        Number of warmup runs before timing (default is 0)
    dtype_c : type
        Data type for the complex grid (e.g. np.complex128)
    dtype_out : type
        Data type for the output (e.g. np.int32)
    kwargs : dict
        Additional keyword arguments to pass to func_calc (e.g. threshold, max_iterations)

    Returns
    -------
    tuple
        (median_time, mean_time, stddev_time, result) where result is the output of func_calc
    """

    # Prepare kwargs for each function
    gen_kwargs = {'dtype': dtype_c} if dtype_c is not None else {}
    calc_kwargs = {**kwargs}
    if dtype_out is not None:
        calc_kwargs['dtype'] = dtype_out

    times = []
    result = None
    for _ in range(warmup_runs):
        C = func_gen(image_size, **gen_kwargs)
        func_calc(C, **calc_kwargs)
    for _ in range(n_runs):
        C = func_gen(image_size, **gen_kwargs)
        t0 = time.perf_counter()
        result = func_calc(C, **calc_kwargs)
        times.append(time.perf_counter() - t0)
    median_t = statistics.median(times)
    mean_t = statistics.mean(times)
    stddev_t = statistics.stdev(times) if len(times) > 1 else 0.0
    print(f"Median: {median_t:.4f} s, Mean: {mean_t:.4f} s, StdDev: {stddev_t:.4f} s "
          f"(min = {min(times):.4f}, max = {max(times):.4f})")
    return median_t, mean_t, stddev_t, result


def mandelbrot_time_test(func_gen, func_calc,
                         start_size_log_2: int = 0, top_size_log_2: int = 5,
                         n_runs_per_size: int = 3,
                         warmup_runs: int = 0,
                         dtype_c=None, dtype_out=None,
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
    warmup_runs : int
        Number of warmup runs before timing (default is 0)
    dtype_c : type
        Data type for the complex grid (e.g. np.complex128)
    dtype_out : type
        Data type for the output (e.g. np.int32)
    show_plots : bool
        Whether to display plots (default is True)
    kwargs : dict
        Additional keyword arguments to pass to func_calc (e.g. threshold, max_iterations)

    Returns
    -------
    tuple
        (results, medians, means, stddevs, image_sizes) where:
        - results: list of computation results for each image size
        - medians: list of median times for each image size
        - means: list of mean times for each image size
        - stddevs: list of standard deviations for each image size
        - image_sizes: list of tested image sizes
    """
    image_sizes = [256 * 2 ** i for i in range(start_size_log_2,
                                               top_size_log_2 + 1)]  # 256, 512, 1024, 2048, 4096, 8192 if top_size_log_2 = 5
    results = []
    medians = []
    means = []
    stddevs = []
    for size in image_sizes:
        print(f"\nImage size {size}x{size}:")
        median, mean, stddev, result = benchmark(func_gen, func_calc, image_size=size, n_runs=n_runs_per_size,
                                                 warmup_runs=warmup_runs,
                                                 dtype_c=dtype_c, dtype_out=dtype_out,
                                                 **kwargs)
        results.append(result)
        medians.append(median)
        means.append(mean)
        stddevs.append(stddev)
        if show_plots:
            plt.plot(image_sizes[:len(medians)], medians, marker='o')
            plt.show()
    return results, medians, means, stddevs, image_sizes
