"""
workgroup_sweep_opencl.py — Level 2 bonus exercise for L10.

Measure how runtime of the Mandelbrot kernel varies with the OpenCL
work-group (local) size. No new kernel code — this exercise is about
*measurement*, not new GPU concepts.

Background:
    When you call a kernel as `prog.mandelbrot(queue, (N,N), None, ...)`,
    the third argument is the local size. `None` lets OpenCL choose for
    you. A specific tuple like `(16, 16)` forces that work-group shape.
    Different local sizes map to compute units differently and can change
    runtime by 2-5x on the same hardware.

Usage:
    python workgroup_sweep_opencl.py

Output:
    Console: one-line-per-local-size timing table.
    File:    workgroup_sweep.png  (bar chart next to this script).
"""

from pathlib import Path
import statistics
import time
import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt


N         = 4096      # image resolution
MAX_ITER  = 200
RUNS      = 5         # median of this many timings per local size

X_MIN, X_MAX = -2.5, 1.0
Y_MIN, Y_MAX = -1.25, 1.25

# Local sizes to try. None = "let OpenCL choose".
# Sizes > device max_work_group_size are skipped at runtime.
LOCAL_SIZES = [
    None,
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 4),
    (4, 64),
    (128, 2),
    (256, 1),
    (1, 256),
]

KERNEL_SRC = """
__kernel void mandelbrot(
    __global int *result,
    const float x_min, const float x_max,
    const float y_min, const float y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0); int row = get_global_id(1);
    if (col >= N || row >= N) return;

    float c_real = x_min + col * (x_max - x_min) / (float)N;
    float c_imag = y_min + row * (y_max - y_min) / (float)N;

    float zr = 0.0f, zi = 0.0f;
    int count = 0;
    while (count < max_iter && zr*zr + zi*zi <= 4.0f) {
        float tmp = zr*zr - zi*zi + c_real;
        zi = 2.0f * zr * zi + c_imag;
        zr = tmp;
        count++;
    }
    result[row * N + col] = count;
}
"""


def time_one(kernel, queue, result_dev, local_size, runs):
    """Return median wall time (seconds) for one local-size choice.

    Returns None if the launch fails (e.g. local size too large for device).
    """
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        try:
            kernel(
                queue, (N, N), local_size, result_dev,
                np.float32(X_MIN), np.float32(X_MAX),
                np.float32(Y_MIN), np.float32(Y_MAX),
                np.int32(N), np.int32(MAX_ITER),
            )
            queue.finish()
        except cl.Error as e:
            return None, str(e)
        times.append(time.perf_counter() - t0)
    return statistics.median(times), None


if __name__ == "__main__":
    ctx   = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    dev   = ctx.devices[0]

    print(f"Device:              {dev.name}")
    print(f"Max work-group size: {dev.max_work_group_size}")
    print(f"Compute units:       {dev.max_compute_units}")
    print(f"Image N:             {N}  max_iter: {MAX_ITER}  runs: {RUNS}\n")

    prog = cl.Program(ctx, KERNEL_SRC).build()
    kernel = cl.Kernel(prog, "mandelbrot")
    result_dev = cl.Buffer(
        ctx, cl.mem_flags.WRITE_ONLY, N * N * np.int32().nbytes
    )

    # Warm up (kernel compile).
    time_one(kernel, queue, result_dev, local_size=None, runs=1)

    results = []
    print(f"{'local_size':>12}  {'median (ms)':>12}  {'rel':>6}")
    print("-" * 40)
    for ls in LOCAL_SIZES:
        t, err = time_one(kernel, queue, result_dev, ls, RUNS)
        label = "auto" if ls is None else f"{ls[0]}x{ls[1]}"
        if t is None:
            print(f"{label:>12}  {'skipped':>12}  ({err.splitlines()[0]})")
            continue
        results.append((label, t))
        print(f"{label:>12}  {t*1e3:12.2f}")

    # Make "rel" column against the fastest.
    fastest = min(t for _, t in results)
    for label, t in results:
        pass  # already printed

    # --- Plot -----------------------------------------------------------
    labels = [r[0] for r in results]
    times_ms = [r[1] * 1e3 for r in results]
    colors = ['#1f4e8c' if lbl != 'auto' else '#b03030' for lbl in labels]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(labels, times_ms, color=colors, edgecolor='white')
    ax.set_ylabel("Median runtime (ms)")
    ax.set_title(
        f"Mandelbrot {N}x{N} — runtime vs work-group size\n"
        f"{dev.name}  ({RUNS} runs, median)",
        fontsize=11,
    )
    ax.grid(axis='y', alpha=0.3)
    # Annotate bar heights
    for bar, t in zip(bars, times_ms):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{t:.1f}", ha='center', va='bottom', fontsize=8.5)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()

    out = Path(__file__).parent / "workgroup_sweep.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out}")
    print(f"Fastest: {min(results, key=lambda r: r[1])[0]}  "
          f"({min(times_ms):.1f} ms)")


"""
Device:              NVIDIA GeForce RTX 5050 Laptop GPU
Max work-group size: 1024
Compute units:       20
Image N:             4096  max_iter: 200  runs: 5

  local_size   median (ms)     rel
----------------------------------------
        auto          1.26
         4x4          3.20
         8x8          1.25
       16x16          1.22
       32x32          2.00
        64x4          1.25
        4x64          1.22
       128x2          1.28
       256x1          1.27
       1x256          1.50

Saved: /home/peppermint/Aalborg/CE8/numerical_scientific_computing/mandelbrot-nsc/gpu_computing/examples/workgroup_sweep.png
Fastest: 4x64  (1.2 ms)
"""
