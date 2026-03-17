import sys
import time
from typing import Tuple

import numpy as np
from numba import njit
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

NUM_RUNS = 10
WARMUP_RUNS = 3


@njit(cache=True)
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


@njit(cache=True)
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
                        num_chunks: int = None,
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
    num_chunks: int
        Number of chunks to divide the work into (default is None, which means equal to num_workers)
    warmup: bool
        Whether to perform a warmup run before timing (default is False)

    Returns
    -------
    np.ndarray
        2D array representing the Mandelbrot set
    float
        Time taken for the parallel computation in seconds
    """
    if num_chunks is None:
        num_chunks = num_workers
    chunk_size = max(1, N // num_chunks)
    chunks, row = [], 0
    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, threshold, max_iter))
        row = row_end
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
        mandelbrot_parallel(None, N, x_min, x_max, y_min, y_max, num_workers=num_workers, warmup=True)
    # Timing the execution
    times = []
    for i in range(NUM_RUNS):
        _, t = mandelbrot_parallel(None, N, x_min, x_max, y_min, y_max, num_workers=num_workers)
        times.append(t)
    median_t = np.median(times)
    mean_t = np.mean(times)
    stddev_t = np.std(times, ddof=1) if len(times) > 1 else 0.0
    print(f"Median: {median_t:.4f} s, Mean: {mean_t:.4f} s, StdDev: {stddev_t:.4f} s "
          f"(min = {min(times):.4f}, max = {max(times):.4f})")


def parallel_benchmark_whole(N: int):
    # Sweep from 1 to cpu_count workers, plot results
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
                 fontsize=15, fontweight='bold', y=0.98)

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
    output_file = f'mandelbrot_speedup_efficiency-res_{N}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as '{output_file}'")

    # Print summary statistics
    print(f"\n{'=' * 70}")
    print(f"Performance Summary ({N}×{N} resolution):")
    print(f"{'=' * 70}")
    print(f"{'Workers':<10} {'Time (s)':<12} {'Speedup':<12} {'Efficiency (%)':<15}")
    print(f"{'-' * 70}")
    for i, nw in enumerate(worker_counts):
        print(f"{nw:<10} {times_array[i]:<12.4f} {speedup[i]:<12.2f} {efficiency[i]:<15.1f}")
    print(f"{'=' * 70}")
    print(f"\nKey Metrics:")
    print(f"  Serial time (1 worker):    {serial_time:.4f}s")
    print(f"  Fastest time ({np.argmin(times_array)+1} workers): {times_array.min():.4f}s")
    print(f"  Best speedup:              {speedup.max():.2f}x")

    plt.show()

    """
    ======================================================================
    Performance Summary (1024×1024 resolution):
    ======================================================================
    Workers    Time (s)     Speedup      Efficiency (%) 
    ----------------------------------------------------------------------
    1          0.0557       1.00         100.0          
    2          0.0274       2.03         101.6          
    3          0.0369       1.51         50.4           
    4          0.0238       2.34         58.5           
    5          0.0259       2.15         43.1           
    6          0.0203       2.75         45.8           
    7          0.0202       2.76         39.4           
    8          0.0167       3.35         41.8           
    9          0.0163       3.43         38.1           
    10         0.0150       3.72         37.2           
    11         0.0140       3.98         36.2           
    12         0.0132       4.22         35.2           
    13         0.0127       4.39         33.8           
    14         0.0120       4.65         33.2           
    15         0.0115       4.84         32.3           
    16         0.0112       4.96         31.0           
    ======================================================================
    
    Key Metrics:
      Serial time (1 worker):    0.0557s
      Fastest time (16 workers): 0.0112s
      Best speedup:              4.96x
    
    ######################################################################
    
    ======================================================================
    Performance Summary (4096×4096 resolution):
    ======================================================================
    Workers    Time (s)     Speedup      Efficiency (%) 
    ----------------------------------------------------------------------
    1          0.9046       1.00         100.0          
    2          0.5036       1.80         89.8           
    3          0.5960       1.52         50.6           
    4          0.3883       2.33         58.2           
    5          0.4131       2.19         43.8           
    6          0.3002       3.01         50.2           
    7          0.2966       3.05         43.6           
    8          0.2481       3.65         45.6           
    9          0.2383       3.80         42.2           
    10         0.2056       4.40         44.0           
    11         0.1993       4.54         41.3           
    12         0.1836       4.93         41.1           
    13         0.1746       5.18         39.8           
    14         0.1614       5.60         40.0           
    15         0.1539       5.88         39.2           
    16         0.1466       6.17         38.6           
    ======================================================================
    
    Key Metrics:
      Serial time (1 worker):    0.9046s
      Fastest time (16 workers): 0.1466s
      Best speedup:              6.17x
    """


def parallel_benchmark_chunkified(N: int):
    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5
    num_workers = cpu_count() # Was actual top speedup in L04
    chunk_multipliers = [1, 2, 4, 8, 16]
    chunk_counts = [m * num_workers for m in chunk_multipliers]
    times = []

    # Measure serial T1 at the same N — needed for LIF
    print("Measuring serial baseline T1...")
    for _ in range(WARMUP_RUNS):
        mandelbrot_serial(N, x_min, x_max, y_min, y_max)
    t1_times = []
    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()
        mandelbrot_serial(N, x_min, x_max, y_min, y_max)
        t1_times.append(time.perf_counter() - t0)
    T1 = np.median(t1_times)
    print(f"  T1 = {T1:.4f}s")

    with Pool(num_workers) as pool:
        # Global warmup: load Numba JIT cache into all workers once
        for i in range(WARMUP_RUNS):
            mandelbrot_parallel(pool, N, x_min, x_max, y_min, y_max,
                                num_workers=num_workers, num_chunks=num_workers, warmup=False)
        for num_chunks in chunk_counts:
            print(f"Running parallel implementation timing for resolution {N}×{N} "
                  f"with {num_workers} workers, {num_chunks} chunks")

            # Per-chunk warmup: workers are already hot, just one call to settle scheduling
            mandelbrot_parallel(pool, N, x_min, x_max, y_min, y_max,
                                num_workers=num_workers, num_chunks=num_chunks, warmup=False)

            run_times = []
            for i in range(NUM_RUNS):
                _, t = mandelbrot_parallel(pool, N, x_min, x_max, y_min, y_max,
                                           num_workers=num_workers, num_chunks=num_chunks, warmup=False)
                run_times.append(t)
            times.append(np.median(run_times))

    chunk_counts_array = np.array(chunk_counts)
    times_array = np.array(times)

    baseline_time = times_array[0]  # 1× chunks (== num_workers chunks)
    speedup = baseline_time / times_array
    ideal_speedup = np.ones_like(times_array)  # ideal is flat (same workers throughout)
    efficiency = (speedup / (chunk_counts_array / num_workers)) * 100
    lif = num_workers * times_array / T1 - 1  # LIF = p*Tp/T1 - 1

    x_labels = [f"{m}×\n({c})" for m, c in zip(chunk_multipliers, chunk_counts)]
    x_pos = list(range(len(chunk_counts)))

    fig, (ax_time, ax_speedup, ax_lif) = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle(
        f'Mandelbrot Chunk Sweep — {num_workers} Workers, {N}×{N} Resolution',
        fontsize=15, fontweight='bold', y=0.98
    )

    # LEFT PLOT: Execution Time
    color_time = '#6A994E'
    ax_time.plot(x_pos, times_array, 'o-', linewidth=2.5, markersize=9,
                 color=color_time, label='Execution Time')
    ax_time.fill_between(x_pos, times_array, alpha=0.3, color=color_time)
    ax_time.set_xlabel('Chunk Multiplier (chunks)', fontsize=12, fontweight='bold')
    ax_time.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold', color=color_time)
    ax_time.tick_params(axis='y', labelcolor=color_time)
    ax_time.set_xticks(x_pos)
    ax_time.set_xticklabels(x_labels)
    ax_time.grid(True, alpha=0.3, linestyle='--')
    ax_time.set_title('Execution Time vs Num Chunks', fontsize=13, fontweight='bold', pad=10)

    for i, (xp, t) in enumerate(zip(x_pos, times_array)):
        ax_time.annotate(f'{t:.3f}s', xy=(xp, t), xytext=(0, 10),
                         textcoords='offset points', ha='center', fontsize=9,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor=color_time, alpha=0.3))
    ax_time.legend(loc='upper right', fontsize=10)

    # MIDDLE PLOT: Relative Speedup and Efficiency
    color_speedup = '#2E86AB'
    ax_speedup.set_xlabel('Chunk Multiplier (chunks)', fontsize=12, fontweight='bold')
    ax_speedup.set_ylabel('Relative Speedup (vs 1×)', fontsize=12, fontweight='bold', color=color_speedup)
    line1 = ax_speedup.plot(x_pos, speedup, 'o-', linewidth=2.5, markersize=9,
                            color=color_speedup, label='Relative Speedup')
    line2 = ax_speedup.plot(x_pos, ideal_speedup, '--', linewidth=2,
                            color='#F18F01', label='Ideal (flat)', alpha=0.7)
    ax_speedup.tick_params(axis='y', labelcolor=color_speedup)
    ax_speedup.set_xticks(x_pos)
    ax_speedup.set_xticklabels(x_labels)
    ax_speedup.grid(True, alpha=0.3, linestyle='--')
    ax_speedup.set_title('Relative Speedup and Efficiency vs Num Chunks',
                         fontsize=13, fontweight='bold', pad=10)

    ax_eff = ax_speedup.twinx()
    color_efficiency = '#C73E1D'
    ax_eff.set_ylabel('Efficiency (%)', fontsize=12, fontweight='bold', color=color_efficiency)
    line3 = ax_eff.plot(x_pos, efficiency, 's-', linewidth=2, markersize=7,
                        color=color_efficiency, label='Efficiency (%)', alpha=0.8)
    ax_eff.tick_params(axis='y', labelcolor=color_efficiency)
    ax_eff.set_ylim([0, 105])

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax_speedup.legend(lines, labels, loc='upper left', fontsize=10)

    # RIGHT PLOT: LIF
    color_lif = '#7B2D8B'
    ax_lif.plot(x_pos, lif, 'o-', linewidth=2.5, markersize=9,
                color=color_lif, label='LIF')
    ax_lif.axhline(y=0, color='#F18F01', linestyle='--', linewidth=2,
                   alpha=0.7, label='Perfect balance (LIF=0)')
    ax_lif.fill_between(x_pos, lif, 0, alpha=0.2, color=color_lif)
    ax_lif.set_xlabel('Chunk Multiplier (chunks)', fontsize=12, fontweight='bold')
    ax_lif.set_ylabel('LIF  (p·Tₚ/T₁ − 1)', fontsize=12, fontweight='bold', color=color_lif)
    ax_lif.tick_params(axis='y', labelcolor=color_lif)
    ax_lif.set_xticks(x_pos)
    ax_lif.set_xticklabels(x_labels)
    ax_lif.grid(True, alpha=0.3, linestyle='--')
    ax_lif.set_title('Load Imbalance Factor vs Num Chunks', fontsize=13, fontweight='bold', pad=10)

    for i, (xp, l) in enumerate(zip(x_pos, lif)):
        ax_lif.annotate(f'{l:.3f}', xy=(xp, l), xytext=(0, 10),
                        textcoords='offset points', ha='center', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=color_lif, alpha=0.3))
    ax_lif.legend(loc='upper right', fontsize=10)

    ##################
    fig.tight_layout()

    output_file = f'mandelbrot_chunk_sweep-res_{N}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as '{output_file}'")

    print(f"\n{'=' * 75}")
    print(f"Chunk Sweep Summary ({N}×{N} resolution; {num_workers} workers):")
    print(f"{'=' * 75}")
    print(f"{'Multiplier':<12} {'Chunks':<10} {'Time (s)':<12} {'Speedup':<12} {'Efficiency (%)':<15}")
    print(f"{'-' * 75}")
    for i, (m, c) in enumerate(zip(chunk_multipliers, chunk_counts)):
        if m < 10:
            print(f"{m}×{'':<10} {c:<10} {times_array[i]:<12.4f} {speedup[i]:<12.2f} {efficiency[i]:<15.1f}")
        else:
            print(f"{m}×{'':<9} {c:<10} {times_array[i]:<12.4f} {speedup[i]:<12.2f} {efficiency[i]:<15.1f}")
    print(f"{'=' * 75}")
    print(f"\nKey Metrics:")
    print(f"  Baseline (1× / {num_workers} chunks): {baseline_time:.4f}s")
    print(f"  Fastest ({chunk_multipliers[np.argmin(times_array)]}× / "
          f"{chunk_counts[np.argmin(times_array)]} chunks): {times_array.min():.4f}s")
    print(f"  Best relative speedup:   {speedup.max():.2f}x  @ {chunk_multipliers[np.argmax(speedup)]}× chunks")
    print(f"  Min LIF (best balance):  {lif.min():.3f}  @ {chunk_multipliers[np.argmin(lif)]}× chunks")

    plt.show()

    """
    ===========================================================================
    Chunk Sweep Summary (1024×1024 resolution; 16 workers):
    ===========================================================================
    Multiplier   Chunks     Time (s)     Speedup      Efficiency (%) 
    ---------------------------------------------------------------------------
    1×           16         0.0116       1.00         100.0          
    2×           32         0.0114       1.01         50.6           
    4×           64         0.0180       0.64         16.1           
    8×           128        0.0166       0.70         8.7            
    16×          256        0.0184       0.63         3.9            
    ===========================================================================
    
    Key Metrics:
      Baseline (1× / 16 chunks): 0.0116s
      Fastest (2× / 32 chunks): 0.0114s
      Best relative speedup:   1.01x  @ 2× chunks
      Min LIF (best balance):  3.130  @ 2× chunks
    
    ###########################################################################
    
    ===========================================================================
    Chunk Sweep Summary (4096×4096 resolution; 16 workers):
    ===========================================================================
    Multiplier   Chunks     Time (s)     Speedup      Efficiency (%) 
    ---------------------------------------------------------------------------
    1×           16         0.1461       1.00         100.0          
    2×           32         0.1034       1.41         70.7           
    4×           64         0.0983       1.49         37.1           
    8×           128        0.1105       1.32         16.5           
    16×          256        0.1021       1.43         8.9            
    ===========================================================================
    
    Key Metrics:
      Baseline (1× / 16 chunks): 0.1461s
      Fastest (4× / 64 chunks): 0.0983s
      Best relative speedup:   1.49x  @ 4× chunks
      Min LIF (best balance):  1.162  @ 4× chunks
    """


if __name__ == "__main__":
    # serial_sanity_check()
    # parallel_sanity_check()
    # parallel_timing()
    parallel_benchmark_whole(1024)
    parallel_benchmark_whole(4096)
    parallel_benchmark_chunkified(1024)
    parallel_benchmark_chunkified(4096)
