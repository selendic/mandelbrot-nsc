"""
iteration_histogram_opencl.py — Level 3 bonus exercise for L10.

First encounter with atomic writes on the GPU. The Mandelbrot kernel
computes an iteration count for each pixel; here we additionally build a
1D histogram counting *how many pixels landed at each iteration value*.

Why it matters:
    - Different work-items finish at different iteration counts, so each
      one has to write into a different histogram bin. This is a "scatter"
      access pattern — the opposite of Mandelbrot's per-pixel write.
    - Without atomics, concurrent increments race and undercount. The fix
      is `atomic_inc(&hist[count])`, which is one line of OpenCL C but a
      big conceptual step.

By-product: plotting the histogram shows where `max_iter` starts to
saturate. A long spike at `max_iter` means many pixels hit the limit —
consider raising max_iter for better interior detail.

This is the "atomics-only" half of the Buddhabrot capstone. Do this
before Buddhabrot if possible.

Usage:
    python iteration_histogram_opencl.py

Output:
    Console: summary stats (mean, saturated pixels).
    Files:   iteration_histogram.png   — the histogram.
             iteration_histogram_image.png — the Mandelbrot it came from.
"""

from pathlib import Path
import time
import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt


N         = 1024
MAX_ITER  = 200
X_MIN, X_MAX = -2.5, 1.0
Y_MIN, Y_MAX = -1.25, 1.25

# ---------------------------------------------------------------------------
# Kernel — same Mandelbrot loop, plus one extra line at the end that does
# an atomic increment on the histogram. Note the histogram is uint so that
# atomic_inc on __global uint* is guaranteed to be in OpenCL 1.2 core.
# ---------------------------------------------------------------------------
KERNEL_SRC = """
__kernel void mandelbrot_hist(
    __global int  *result,
    __global uint *hist,
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

    // Scatter write: different pixels finish at different counts, so every
    // work-item targets a different bin. Atomic avoids lost updates when
    // two work-items happen to land on the same bin.
    atomic_inc(&hist[count]);
}
"""


def run(ctx, queue, N, max_iter):
    prog = cl.Program(ctx, KERNEL_SRC).build()

    image = np.zeros((N, N), dtype=np.int32)
    hist  = np.zeros(max_iter + 1, dtype=np.uint32)

    image_dev = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, image.nbytes)
    hist_dev  = cl.Buffer(
        ctx,
        cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
        hostbuf=hist,
    )

    t0 = time.perf_counter()
    prog.mandelbrot_hist(
        queue, (N, N), None,
        image_dev, hist_dev,
        np.float32(X_MIN), np.float32(X_MAX),
        np.float32(Y_MIN), np.float32(Y_MAX),
        np.int32(N), np.int32(max_iter),
    )
    queue.finish()
    elapsed = time.perf_counter() - t0

    cl.enqueue_copy(queue, image, image_dev)
    cl.enqueue_copy(queue, hist,  hist_dev)
    queue.finish()
    return image, hist, elapsed


if __name__ == "__main__":
    ctx   = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    dev   = ctx.devices[0]
    print(f"Device: {dev.name}\n")

    # Warm-up.
    _, _, _ = run(ctx, queue, N=64, max_iter=MAX_ITER)

    image, hist, elapsed = run(ctx, queue, N, MAX_ITER)
    total = N * N

    # Sanity check: sum of histogram must equal number of pixels.
    assert hist.sum() == total, (
        f"histogram sum {hist.sum()} != total pixels {total} — "
        f"atomics are likely missing or broken"
    )

    saturated = int(hist[MAX_ITER])
    print(f"Image:      {N}x{N}  max_iter: {MAX_ITER}")
    print(f"Elapsed:    {elapsed*1e3:.1f} ms")
    print(f"Histogram:  sum = {hist.sum():,}  (matches {total} pixels)")
    print(f"Saturated:  {saturated:,} pixels hit max_iter "
          f"({100*saturated/total:.1f}%)")
    print(f"Mean iter:  {(hist * np.arange(MAX_ITER + 1)).sum() / total:.1f}")

    # --- Plot histogram -------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(np.arange(MAX_ITER + 1), hist, width=1.0,
           color='#1f4e8c', edgecolor='none')
    # Highlight the saturation bin in red.
    ax.bar([MAX_ITER], [hist[MAX_ITER]], width=1.0, color='#b03030',
           label=f'max_iter ({saturated:,} pixels)')
    ax.set_yscale('log')
    ax.set_xlabel("Iteration count at escape")
    ax.set_ylabel("Number of pixels (log scale)")
    ax.set_title(
        f"Mandelbrot iteration-count histogram  "
        f"({N}x{N}, max_iter={MAX_ITER})",
        fontsize=11,
    )
    ax.legend(loc='upper center')
    ax.grid(axis='y', alpha=0.3, which='both')
    plt.tight_layout()
    out_hist = Path(__file__).parent / "iteration_histogram.png"
    plt.savefig(out_hist, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_hist}")

    # --- Plot the Mandelbrot itself for reference ------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(image, cmap='hot', origin='lower')
    ax.set_title(f"Mandelbrot (source of the histogram)", fontsize=11)
    ax.axis('off')
    plt.tight_layout()
    out_img = Path(__file__).parent / "iteration_histogram_image.png"
    plt.savefig(out_img, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_img}")


"""
Device: NVIDIA GeForce RTX 5050 Laptop GPU
Image:      1024x1024  max_iter: 200
Elapsed:    0.8 ms
Histogram:  sum = 1,048,576  (matches 1048576 pixels)
Saturated:  183,036 pixels hit max_iter (17.5%)
Mean iter:  39.0
"""
