"""
buddhabrot_simple_opencl.py — A minimal Buddhabrot on the GPU.

Written for readability — it's the reference companion to
buddhabrot_tutorial.md.

Design choices (in service of simplicity):
  - One work-item = one sample. No inner loop, no batching.
  - One flat 1D NDRange. No work-group tuning.
  - Inline xorshift32 — the shortest decent PRNG that fits in three lines.
  - Square sampling and view windows, same rectangle for both.
  - No quantile stretching at render time; just a log to tame the dynamic
    range.

The kernel mirrors the plain Mandelbrot kernel you already know, with
exactly two things added:
  1. A per-work-item random number generator, to sample c.
  2. An atomic_inc on a shared global histogram, to record trajectories
     without losing updates when two work-items land on the same pixel.

Read the kernel top-to-bottom. Comments flag the two new ideas.
"""

from pathlib import Path
import time
import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N_SAMPLES = 1 << 24   # 16,777,216 samples — each is one work-item.
MAX_ITER  = 500
IMAGE_N   = 800       # histogram is IMAGE_N x IMAGE_N

# Same rectangle used for sampling c and for the final image view.
X_MIN, X_MAX = -2.0, 1.0
Y_MIN, Y_MAX = -1.5, 1.5


# ---------------------------------------------------------------------------
# Kernel — read this top-to-bottom
# ---------------------------------------------------------------------------
KERNEL_SRC = """
// --- New idea #1: a per-work-item PRNG ---------------------------------
// xorshift32: cheap, 3 lines, good enough for visual Monte Carlo.
// Each work-item carries its own 'state' so parallel streams don't collide.
inline uint xorshift32(uint *state) {
    uint x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

// Map a random uint32 to a float in [0, 1).
inline float rand01(uint *state) {
    return (float)xorshift32(state) * (1.0f / 4294967296.0f);
}

__kernel void buddhabrot(
    __global uint *hist,              // IMAGE_N * IMAGE_N histogram
    const int   N,                    // image side length
    const float x_min, const float x_max,
    const float y_min, const float y_max,
    const int   max_iter,
    const uint  base_seed)
{
    int gid = get_global_id(0);

    // Seed the PRNG so every work-item gets a different stream.
    // (Multiplying by a large odd constant spreads nearby gids apart.)
    uint rng = base_seed ^ ((uint)gid * 2654435761u);
    if (rng == 0u) rng = 1u;          // xorshift can't start from zero

    // --- Step 1: pick a random complex point c ---------------------------
    float c_real = x_min + rand01(&rng) * (x_max - x_min);
    float c_imag = y_min + rand01(&rng) * (y_max - y_min);

    // --- Step 2: does c escape?  (standard Mandelbrot loop) --------------
    float zr = 0.0f, zi = 0.0f;
    int   iter = 0;
    while (iter < max_iter && zr*zr + zi*zi <= 4.0f) {
        float tmp = zr*zr - zi*zi + c_real;
        zi = 2.0f * zr * zi + c_imag;
        zr = tmp;
        iter++;
    }

    // c is in the set → its trajectory is uninformative, skip.
    if (iter >= max_iter) return;

    // --- Step 3: re-iterate and record the trajectory --------------------
    // We know c escapes in 'iter' steps. Replay that orbit and drop a hit
    // into the histogram bin each z_k lands in.
    zr = 0.0f; zi = 0.0f;
    for (int k = 0; k < iter; k++) {
        float tmp = zr*zr - zi*zi + c_real;
        zi = 2.0f * zr * zi + c_imag;
        zr = tmp;

        // Map (zr, zi) to pixel coordinates. Skip if the orbit wandered
        // outside the view window.
        if (zr < x_min || zr >= x_max || zi < y_min || zi >= y_max) continue;
        int col = (int)((zr - x_min) / (x_max - x_min) * (float)N);
        int row = (int)((zi - y_min) / (y_max - y_min) * (float)N);

        // --- New idea #2: atomic scatter write ---------------------------
        // Many work-items may try to increment the same bin at once.
        // atomic_inc serialises per-bin, so no increments are lost.
        atomic_inc(&hist[row * N + col]);
    }
}
"""


def run(ctx, queue):
    prog = cl.Program(ctx, KERNEL_SRC).build()

    hist = np.zeros((IMAGE_N, IMAGE_N), dtype=np.uint32)
    hist_dev = cl.Buffer(
        ctx,
        cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
        hostbuf=hist,
    )

    t0 = time.perf_counter()
    prog.buddhabrot(
        queue, (N_SAMPLES,), None,
        hist_dev,
        np.int32(IMAGE_N),
        np.float32(X_MIN), np.float32(X_MAX),
        np.float32(Y_MIN), np.float32(Y_MAX),
        np.int32(MAX_ITER),
        np.uint32(0xC0FFEE),
    )
    queue.finish()
    elapsed = time.perf_counter() - t0

    cl.enqueue_copy(queue, hist, hist_dev)
    queue.finish()
    return hist, elapsed


if __name__ == "__main__":
    ctx   = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    print(f"Device:   {ctx.devices[0].name}")
    print(f"Samples:  {N_SAMPLES:,}   max_iter: {MAX_ITER}   image: {IMAGE_N}x{IMAGE_N}")

    hist, elapsed = run(ctx, queue)
    rate = N_SAMPLES / elapsed / 1e6
    print(f"Elapsed:  {elapsed:.2f} s   ({rate:.1f} M samples/s)")
    print(f"Hist:     sum={hist.sum():,}   max_bin={hist.max():,}")

    # log(1+x) compresses the dynamic range so filaments stay visible.
    img = np.log1p(hist.astype(np.float32))

    plt.figure(figsize=(6, 6))
    # .T for the classic 'seated Buddha' orientation (imaginary axis horizontal).
    plt.imshow(img.T, cmap='hot', origin='lower')
    plt.title(
        f"Buddhabrot (simple) — {N_SAMPLES/1e6:.0f}M samples, {elapsed:.1f}s",
        fontsize=10,
    )
    plt.axis('off')
    plt.tight_layout()

    out = Path(__file__).parent / "buddhabrot_simple.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved:    {out}")
