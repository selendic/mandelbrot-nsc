import sys
import time
from typing import Tuple

import numpy as np
from numba import njit
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

NUM_RUNS = 10
WARMUP_RUNS = 3


@njit
def mandelbrot_point(c: complex, t: float, max_iter=100):
    """
    Calculates the number of iterations for a point c to escape the Mandelbrot set (numba-njit optimized).

    Parameters
    ----------
    c : complex
        Complex point to evaluate
    t : float
        Escape threshold squared (to avoid computing square root)
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


@njit
def mandelbrot_chunk(row_start: int, row_end: int, N: int,
                     x_min: float, x_max: float, y_min: float, y_max: float,
                     threshold=2, max_iter=100) -> np.ndarray:
    """
    Loops over rows [row start, row end) and all columns.
    Computes pixel coordinates from index + bounds — no arrays received as input.
    Returns a (row end - row start)×N int32 array.

    Parameters
    ----------
    row_start : int
        Starting row index (inclusive)
    row_end : int
        Ending row index (exclusive)
    N : int
        Number of columns (width of the image)
    x_min : float
        Minimum x-coordinate (real part)
    x_max : float
        Maximum x-coordinate (real part)
    y_min : float
        Minimum y-coordinate (imaginary part)
    y_max : float
        Maximum y-coordinate (imaginary part)
    threshold : float
        Escape threshold (default is 2)
    max_iter : int
        Maximum number of iterations (default is 100)

    Returns
    -------
    np.ndarray
        2D array representing the Mandelbrot set
    """
    height = row_end - row_start
    result = np.empty((height, N), dtype=np.int32)
    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N
    t2 = threshold * threshold
    for r in range(height):
        y = y_min + (row_start + r) * dy
        for j in range(N):
            x = x_min + j * dx
            c = complex(x, y)
            result[r, j] = mandelbrot_point(c, t2, max_iter)
    return result


def mandelbrot_serial(N: int, x_min: float, x_max: float, y_min: float, y_max: float,
                      threshold: float = 2, max_iter: int = 100) -> np.ndarray:
    """
    Thin wrapper: calls mandelbrot chunk(0, N, ...) — the whole grid as one chunk.

    Parameters
    ----------
    N: int
        Number of columns (width of the image)
    x_min: float
        Minimum x-coordinate (real part)
    x_max: float
        Maximum x-coordinate (real part)
    y_min: float
        Minimum y-coordinate (imaginary part)
    y_max: float
        Maximum y-coordinate (imaginary part)
    threshold: float
        Escape threshold (default is 2)
    max_iter: int
        Maximum number of iterations (default is 100)

    Returns
    -------
    np.ndarray
        2D array representing the Mandelbrot set
    """
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, threshold, max_iter)


def serial_sanity_check():
    N = 1024
    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5
    print(f"Running sanity check for resolution {N}×{N} (should match full numba implementation)")
    # Warmup runs to compile the Numba functions
    for i in range(WARMUP_RUNS):
        mandelbrot_serial(N, x_min, x_max, y_min, y_max)
    # Timing the execution
    times = []
    for i in range(NUM_RUNS):
        t0 = time.perf_counter()
        mandelbrot_serial(N, x_min, x_max, y_min, y_max)
        times.append(time.perf_counter() - t0)
    median_t = np.median(times)
    mean_t = np.mean(times)
    stddev_t = np.std(times, ddof=1) if len(times) > 1 else 0.0
    print(f"Median: {median_t:.4f} s, Mean: {mean_t:.4f} s, StdDev: {stddev_t:.4f} s "
          f"(min = {min(times):.4f}, max = {max(times):.4f})")

    """
    Running sanity check for resolution 1024×1024 (should match full numba implementation)
    Median: 0.0439 s, Mean: 0.0440 s, StdDev: 0.0004 s (min = 0.0439, max = 0.0452)
    __
    Nice! Well, close enough...
    (Numba Full: 0.0419 ± 0.0009)
    """


def mandelbrot_parallel(pool: Pool, N: int,
                        x_min: float, x_max: float, y_min: float, y_max: float,
                        threshold: float = 2, max_iter: int = 100, num_workers: int = 4,
                        warmup: bool = False) -> Tuple[np.ndarray, float]:
    """
    Builds a list of chunk tuples: (row start, row end, N, x min, x max, y min, y max, max iter).
    Uses pool.map(worker, chunks) to distribute across workers.
    Reassembles with np.vstack(parts).

    Parameters
    ----------
    pool: Pool
        Multiprocessing pool to use for parallel computation
    N: int
        Number of columns (width of the image)
    x_min: float
        Minimum x-coordinate (real part)
    x_max: float
        Maximum x-coordinate (real part)
    y_min: float
        Minimum y-coordinate (imaginary part)
    y_max: float
        Maximum y-coordinate (imaginary part)
    threshold: float
        Escape threshold (default is 2)
    max_iter: int
        Maximum number of iterations (default is 100)
    num_workers: int
        Number of worker processes to use for parallel computation (default is 4)
    warmup: bool
        Whether to perform a warmup run before timing (default is False)

    Returns
    -------
    np.ndarray
        2D array representing the Mandelbrot set
    float
        Time taken for the parallel computation in seconds
    """
    chunk_size = max(1, N // num_workers)
    chunks = []
    for i in range(num_workers):
        row_start = i * chunk_size
        row_end = (i + 1) * chunk_size if i < num_workers - 1 else N
        chunks.append((row_start, row_end, N, x_min, x_max, y_min, y_max, threshold, max_iter))
    if pool is None:
        with Pool(num_workers) as pool:
            # Warmup
            if warmup:
                pool.starmap(mandelbrot_chunk, chunks)
            # Timing
            t = time.perf_counter()
            parts = pool.starmap(mandelbrot_chunk, chunks)
            t = time.perf_counter() - t
    else:
        # Warmup
        if warmup:
            pool.starmap(mandelbrot_chunk, chunks)
        # Timing
        t = time.perf_counter()
        parts = pool.starmap(mandelbrot_chunk, chunks)
        t = time.perf_counter() - t
    return np.vstack(parts), t


def parallel_sanity_check():
    # Compare the output of the parallel implementation to the serial one for a few resolutions
    Ns = [256, 512, 1024]
    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5
    for N in Ns:
        serial_result = mandelbrot_serial(N, x_min, x_max, y_min, y_max)
        parallel_result, _ = mandelbrot_parallel(N, x_min, x_max, y_min, y_max, warmup=True)
        if np.array_equal(serial_result, parallel_result):
            print(f"Sanity check passed for resolution {N}×{N}")
        else:
            print(f"Sanity check failed for resolution {N}×{N}!", file=sys.stderr)

    """
    Sanity check passed for resolution 256×256
    Sanity check passed for resolution 512×512
    Sanity check passed for resolution 1024×1024
    """


def parallel_timing():
    N = 1024
    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5
    num_workers = 4
    print(f"Running parallel implementation timing for resolution {N}×{N} with {num_workers} workers")
    # Do warmup outside
    for i in range(WARMUP_RUNS):
        mandelbrot_parallel(N, x_min, x_max, y_min, y_max, num_workers=num_workers, warmup=True)
    # Timing the execution
    times = []
    for i in range(NUM_RUNS):
        _, t = mandelbrot_parallel(N, x_min, x_max, y_min, y_max, num_workers=num_workers)
        times.append(t)
    median_t = np.median(times)
    mean_t = np.mean(times)
    stddev_t = np.std(times, ddof=1) if len(times) > 1 else 0.0
    print(f"Median: {median_t:.4f} s, Mean: {mean_t:.4f} s, StdDev: {stddev_t:.4f} s "
          f"(min = {min(times):.4f}, max = {max(times):.4f})")


def parallel_benchmark():
    # Sweep from 1 to cpu_count workers, plot results
    N = 1024
    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5
    worker_counts = list(range(1, cpu_count() + 1))
    times = []
    for num_workers in worker_counts:
        print(f"Running parallel implementation timing for resolution {N}×{N} with {num_workers} workers")
        # Initialize the pool once per worker count to avoid overhead in each run
        with Pool(num_workers) as pool:
            # Do warmup outside
            for i in range(WARMUP_RUNS):
                mandelbrot_parallel(pool, N, x_min, x_max, y_min, y_max, num_workers=num_workers, warmup=False)
            # Timing the execution
            run_times = []
            for i in range(NUM_RUNS):
                _, t = mandelbrot_parallel(pool, N, x_min, x_max, y_min, y_max, num_workers=num_workers)
                run_times.append(t)
        times.append(np.median(run_times))

    # Convert to numpy arrays for calculations
    worker_counts_array = np.array(worker_counts)
    times_array = np.array(times)

    # Calculate speedup and efficiency
    serial_time = times_array[0]  # Time with 1 worker
    speedup = serial_time / times_array
    ideal_speedup = worker_counts_array
    efficiency = (speedup / worker_counts_array) * 100  # Efficiency as percentage

    # Create figure with two subplots: one for times, one for speedup/efficiency
    fig, (ax_time, ax_speedup) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Mandelbrot Parallel Performance Analysis ({N}×{N} Resolution)',
                 fontsize=15, fontweight='bold', y=1.02)

    # LEFT PLOT: Execution Time
    color_time = '#6A994E'
    ax_time.plot(worker_counts, times_array, 'o-', linewidth=2.5, markersize=9,
                 color=color_time, label='Execution Time')
    ax_time.fill_between(worker_counts, times_array, alpha=0.3, color=color_time)
    ax_time.set_xlabel('Number of Workers', fontsize=12, fontweight='bold')
    ax_time.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold', color=color_time)
    ax_time.tick_params(axis='y', labelcolor=color_time)
    ax_time.set_xticks(worker_counts)
    ax_time.grid(True, alpha=0.3, linestyle='--')
    ax_time.set_title('Execution Time vs Workers', fontsize=13, fontweight='bold', pad=10)

    # Add time annotations on the plot
    for i, (w, t) in enumerate(zip(worker_counts, times_array)):
        if i % 2 == 0 or i == len(worker_counts) - 1:  # Annotate every other point
            ax_time.annotate(f'{t:.3f}s', xy=(w, t), xytext=(0, 10),
                           textcoords='offset points', ha='center', fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=color_time, alpha=0.3))

    ax_time.legend(loc='upper right', fontsize=10)

    # RIGHT PLOT: Speedup and Efficiency on same axes
    color_speedup = '#2E86AB'
    ax_speedup.set_xlabel('Number of Workers', fontsize=12, fontweight='bold')
    ax_speedup.set_ylabel('Speedup', fontsize=12, fontweight='bold', color=color_speedup)
    line1 = ax_speedup.plot(worker_counts, speedup, 'o-', linewidth=2.5, markersize=9,
                            color=color_speedup, label='Actual Speedup')
    line2 = ax_speedup.plot(worker_counts, ideal_speedup, '--', linewidth=2,
                            color='#F18F01', label='Ideal Speedup', alpha=0.7)
    ax_speedup.tick_params(axis='y', labelcolor=color_speedup)
    ax_speedup.set_xticks(worker_counts)
    ax_speedup.grid(True, alpha=0.3, linestyle='--')
    ax_speedup.set_title('Speedup and Efficiency vs Workers', fontsize=13, fontweight='bold', pad=10)

    # Create second y-axis for efficiency
    ax_eff = ax_speedup.twinx()
    color_efficiency = '#C73E1D'
    ax_eff.set_ylabel('Efficiency (%)', fontsize=12, fontweight='bold', color=color_efficiency)
    line3 = ax_eff.plot(worker_counts, efficiency, 's-', linewidth=2, markersize=7,
                        color=color_efficiency, label='Efficiency (%)', alpha=0.8)
    ax_eff.tick_params(axis='y', labelcolor=color_efficiency)
    ax_eff.set_ylim([0, 105])  # Efficiency from 0-105%

    # Combine legends from both axes
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax_speedup.legend(lines, labels, loc='upper left', fontsize=10)

    fig.tight_layout()

    # Save the plot
    output_file = 'mandelbrot_speedup_efficiency.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved as '{output_file}'")

    # Print summary statistics
    print(f"\n{'=' * 70}")
    print(f"Performance Summary:")
    print(f"{'=' * 70}")
    print(f"{'Workers':<10} {'Time (s)':<12} {'Speedup':<12} {'Efficiency (%)':<15}")
    print(f"{'-' * 70}")
    for i, nw in enumerate(worker_counts):
        print(f"{nw:<10} {times_array[i]:<12.4f} {speedup[i]:<12.2f} {efficiency[i]:<15.1f}")
    print(f"{'=' * 70}")
    print(f"\nKey Metrics:")
    print(f"  Serial time (1 worker):  {serial_time:.4f}s")
    print(f"  Fastest time ({np.argmin(times_array)+1} workers): {times_array.min():.4f}s")
    print(f"  Best speedup:            {speedup.max():.2f}x")
    print(f"  Best efficiency:         {efficiency.max():.1f}%")
    print(f"  Time reduction:          {((serial_time - times_array.min()) / serial_time * 100):.1f}%\n")

    plt.show()


if __name__ == "__main__":
    # serial_sanity_check()
    # parallel_sanity_check()
    # parallel_timing()
    parallel_benchmark()
