import os
import random
import statistics
import time
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np

NUM_SAMPLES = 10_000_000
NUM_RUNS = 10


def estimate_pi_chunk(num_samples):
    inside_circle = 0
    for _ in range(num_samples):
        x, y = random.random(), random.random()
        if x * x + y * y <= 1:
            inside_circle += 1
    return inside_circle


def estimate_pi_parallel(total_num_samples, num_processes=4, chunk_size=None):
    if total_num_samples <= 0:
        raise ValueError("total_num_samples must be > 0")
    if num_processes <= 0:
        raise ValueError("num_processes must be > 0")

    if chunk_size is None:
        chunk_size = max(1, total_num_samples // num_processes)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    num_full_chunks, remainder = divmod(total_num_samples, chunk_size)
    tasks = [chunk_size] * num_full_chunks
    if remainder:
        tasks.append(remainder)

    results, t0 = [], time.perf_counter()
    if num_processes == 1:
        t0 = time.perf_counter()
        results = [estimate_pi_chunk(t) for t in tasks]
    else:
        with Pool(processes=num_processes) as pool:
            t0 = time.perf_counter()
            results = pool.map(estimate_pi_chunk, tasks)
    return 4 * sum(results) / total_num_samples, time.perf_counter() - t0


def non_chunked_version():
    print(f"Num samples: {NUM_SAMPLES}")
    print(f"Num runs:    {NUM_RUNS}\n")

    # Collect data for plotting
    num_workers_list = []
    median_times = []
    mean_estimates = []
    std_estimates = []

    for num_proc in range(1, os.cpu_count() + 1):
        pi_estimates = []
        times = []
        for _ in range(NUM_RUNS):
            pi_estimate, elapsed = estimate_pi_parallel(NUM_SAMPLES, num_proc)
            pi_estimates.append(pi_estimate)
            times.append(elapsed)
        t_serial_med = statistics.median(times)
        pi_estimate_mean = statistics.mean(pi_estimates)
        pi_estimate_std = statistics.stdev(pi_estimates)

        # Store for plotting
        num_workers_list.append(num_proc)
        median_times.append(t_serial_med)
        mean_estimates.append(pi_estimate_mean)
        std_estimates.append(pi_estimate_std)

        print(f"Num workers: {num_proc:2d}")
        print(f"Serial time: {t_serial_med:.3f}s")
        print(f"π estimate: {pi_estimate_mean:.6f}±{pi_estimate_std:.6f}\n")

    # Create comprehensive plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Monte Carlo Pi Estimation - Parallel Performance Analysis', fontsize=16, fontweight='bold')

    # Convert to numpy arrays for easier manipulation
    num_workers = np.array(num_workers_list)
    times = np.array(median_times)
    pi_means = np.array(mean_estimates)
    pi_stds = np.array(std_estimates)

    # Plot 1: Execution Time vs Number of Workers
    ax1 = axes[0, 0]
    ax1.plot(num_workers, times, 'o-', linewidth=2, markersize=8, color='#2E86AB', label='Median Time')
    ax1.fill_between(num_workers, times * 0.95, times * 1.05, alpha=0.2, color='#2E86AB')
    ax1.set_xlabel('Number of Workers', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Execution Time vs Workers', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(num_workers)
    ax1.legend()

    # Plot 2: Speedup and Efficiency
    ax2 = axes[0, 1]
    speedup = times[0] / times
    ideal_speedup = num_workers  # Monte Carlo is embarrassingly parallel, so P=100%
    efficiency = (speedup / num_workers) * 100

    ax2_twin = ax2.twinx()
    line1 = ax2.plot(num_workers, speedup, 'o-', linewidth=2, markersize=8,
                     color='#A23B72', label='Actual Speedup')
    line2 = ax2.plot(num_workers, ideal_speedup, '--', linewidth=2,
                     color='#F18F01', label='Ideal Speedup', alpha=0.7)
    line3 = ax2_twin.plot(num_workers, efficiency, 's-', linewidth=2, markersize=6,
                          color='#C73E1D', label='Efficiency (%)', alpha=0.8)

    ax2.set_xlabel('Number of Workers', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Speedup', fontsize=12, fontweight='bold')
    ax2_twin.set_ylabel('Efficiency (%)', fontsize=12, fontweight='bold', color='#C73E1D')
    ax2_twin.tick_params(axis='y', labelcolor='#C73E1D')
    ax2.set_title('Speedup and Parallel Efficiency', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xticks(num_workers)

    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left')

    # Plot 3: Pi Estimate Accuracy
    ax3 = axes[1, 0]
    actual_pi = np.pi
    pi_error = np.abs(pi_means - actual_pi)

    ax3.errorbar(num_workers, pi_means, yerr=pi_stds, fmt='o-', linewidth=2,
                 markersize=8, capsize=5, capthick=2, color='#6A994E', label='Estimated π')
    ax3.axhline(y=actual_pi, color='#BC4749', linestyle='--', linewidth=2,
                label=f'Actual π = {actual_pi:.6f}', alpha=0.8)
    ax3.fill_between(num_workers, actual_pi - 0.0001, actual_pi + 0.0001,
                     alpha=0.2, color='#BC4749')
    ax3.set_xlabel('Number of Workers', fontsize=12, fontweight='bold')
    ax3.set_ylabel('π Estimate', fontsize=12, fontweight='bold')
    ax3.set_title('π Estimate vs Workers', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xticks(num_workers)
    ax3.legend()
    ax3.set_ylim([min(pi_means) - 0.0005, max(pi_means) + 0.0005])

    # Plot 4: Summary Statistics Table
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Back-solved implied serial fraction (Amdahl's Law)
    best_speedup = speedup.max()
    best_num_workers = num_workers[speedup.argmax()]
    serial_fraction = (1 / best_speedup - 1 / best_num_workers) / (1 - 1 / best_num_workers)

    # Create summary table
    summary_data = []
    summary_data.append(['Metric', 'Value'])
    summary_data.append(['Samples per run', f'{NUM_SAMPLES:,}'])
    summary_data.append(['Number of runs', f'{NUM_RUNS}'])
    summary_data.append(['Available CPUs (log)', f'{os.cpu_count()}'])
    summary_data.append(['─' * 25, '─' * 15])
    summary_data.append(['Best speedup', f'{best_speedup:.2f}x'])
    summary_data.append(['Num workers for best speedup', f'{best_num_workers}'])
    summary_data.append(['Implied serial fraction', f'{serial_fraction:.2%}'])
    summary_data.append(['Serial time', f'{times[0]:.3f}s'])
    summary_data.append(['Fastest time', f'{times.min():.3f}s'])

    table = ax4.table(cellText=summary_data, cellLoc='left', loc='center',
                      colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style the table
    for i, key in enumerate(summary_data):
        if i == 0:  # Header
            table[(i, 0)].set_facecolor('#2E86AB')
            table[(i, 1)].set_facecolor('#2E86AB')
            table[(i, 0)].set_text_props(weight='bold', color='white')
            table[(i, 1)].set_text_props(weight='bold', color='white')
        elif '─' in key[0]:  # Separator
            table[(i, 0)].set_facecolor('#E5E5E5')
            table[(i, 1)].set_facecolor('#E5E5E5')
        else:
            if i % 2 == 0:
                table[(i, 0)].set_facecolor('#F5F5F5')
                table[(i, 1)].set_facecolor('#F5F5F5')

    # ax4.set_title('Performance Summary', fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()

    # Save the plot
    output_file = 'monte_carlo_parallel_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as '{output_file}'")

    plt.show()

    """
    Num samples: 10000000
    Num runs:    10

    Num workers:  1
    Serial time: 0.732s
    PI estimate: 3.141915±0.000455

    Num workers:  2
    Serial time: 0.387s
    PI estimate: 3.141359±0.000412

    Num workers:  3
    Serial time: 0.278s
    PI estimate: 3.141513±0.000419

    Num workers:  4
    Serial time: 0.226s
    PI estimate: 3.141872±0.000546

    Num workers:  5
    Serial time: 0.194s
    PI estimate: 3.141560±0.000413

    Num workers:  6
    Serial time: 0.176s
    PI estimate: 3.141530±0.000574

    Num workers:  7
    Serial time: 0.163s
    PI estimate: 3.141641±0.000592

    Num workers:  8
    Serial time: 0.157s
    PI estimate: 3.141517±0.000413

    Num workers:  9
    Serial time: 0.189s
    PI estimate: 3.141682±0.000436

    Num workers: 10
    Serial time: 0.169s
    PI estimate: 3.141268±0.000475

    Num workers: 11
    Serial time: 0.174s
    PI estimate: 3.141539±0.000515

    Num workers: 12
    Serial time: 0.176s
    PI estimate: 3.141716±0.000569

    Num workers: 13
    Serial time: 0.171s
    PI estimate: 3.141258±0.000659

    Num workers: 14
    Serial time: 0.169s
    PI estimate: 3.141375±0.000398

    Num workers: 15
    Serial time: 0.163s
    PI estimate: 3.141386±0.000442

    Num workers: 16
    Serial time: 0.162s
    PI estimate: 3.141786±0.000643
    """


def chunked_version():
    # Print and investigate to find optimal chunk size for my machine
    total_work = 10_000_000
    n_proc = os.cpu_count() // 2
    chunk_sizes = [10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    num_iter_per_chunk_size = 10

    print("Total work: ", total_work)
    est_pis = {chunk_size: [] for chunk_size in chunk_sizes}
    times = {chunk_size: [] for chunk_size in chunk_sizes}
    for chunk_size in chunk_sizes:
        for i in range(num_iter_per_chunk_size):
            est_pi, t = estimate_pi_parallel(total_work, num_processes=n_proc, chunk_size=chunk_size)
            est_pis[chunk_size].append(est_pi)
            times[chunk_size].append(t)
    for chunk_size in chunk_sizes:
        print(f"Chunk size: {chunk_size:>8} | π estimate (med): {statistics.median(est_pis[chunk_size]):.6f} | Time (med): {statistics.mean(times[chunk_size]):.3f} ± {statistics.stdev(times[chunk_size]):.3f}s")

    """
    Total work:  10000000
    Chunk size:       10 | π estimate (med): 3.141596 | Time (med): 0.133 ± 0.007s
    Chunk size:      100 | π estimate (med): 3.141559 | Time (med): 0.106 ± 0.003s <--
    Chunk size:     1000 | π estimate (med): 3.141715 | Time (med): 0.111 ± 0.004s
    Chunk size:    10000 | π estimate (med): 3.141645 | Time (med): 0.112 ± 0.002s
    Chunk size:   100000 | π estimate (med): 3.141562 | Time (med): 0.130 ± 0.001s
    Chunk size:  1000000 | π estimate (med): 3.141760 | Time (med): 0.160 ± 0.002s
    Chunk size: 10000000 | π estimate (med): 3.141652 | Time (med): 0.750 ± 0.034s
    """


if __name__ == "__main__":
    # non_chunked_version()
    chunked_version()
