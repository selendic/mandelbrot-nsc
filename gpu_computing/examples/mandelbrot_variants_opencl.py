"""
mandelbrot_variants_opencl.py — Level 1 bonus exercises for L10.

Three escape-time fractals that reuse almost everything from the main
Mandelbrot kernel:

  Julia set       — z_{n+1} = z_n^2 + c, c fixed, z_0 = pixel.
                    Same formula as Mandelbrot; what changes is what's constant
                    and what's the starting point.

  Burning Ship    — z_{n+1} = (|Re(z)| + i|Im(z)|)^2 + c.
                    One extra line: take absolute value of z before squaring.

  Tricorn         — z_{n+1} = conj(z)^2 + c.
                    One sign flip: the imaginary update term becomes negative.

Usage:
    python mandelbrot_variants_opencl.py

Output:
    Console: per-variant timing.
    File:    mandelbrot_variants.png  (2x2 panel next to this script).
"""

from pathlib import Path
import time
import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Per-fractal view windows (chosen to show each one nicely)
# ---------------------------------------------------------------------------
VIEWS = {
    "Mandelbrot":   (-2.5,  1.0, -1.25, 1.25),
    "Julia":        (-1.5,  1.5, -1.0,  1.0 ),
    "Burning Ship": (-2.2,  1.3, -2.0,  1.0 ),
    "Tricorn":      (-2.2,  1.5, -1.5,  1.5 ),
}
MAX_ITER = 200
JULIA_C  = (-0.7, 0.27015)   # classic dendrite-shaped Julia set


# ---------------------------------------------------------------------------
# Kernels — one per variant. Structurally identical to mandelbrot_opencl.py;
# only the inner update rule differs. Compare them side-by-side.
# ---------------------------------------------------------------------------
KERNELS = {}

KERNELS["Mandelbrot"] = """
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

# Julia — c is a constant parameter, z_0 comes from the pixel.
KERNELS["Julia"] = """
__kernel void julia(
    __global int *result,
    const float x_min, const float x_max,
    const float y_min, const float y_max,
    const float c_real, const float c_imag,
    const int N, const int max_iter)
{
    int col = get_global_id(0); int row = get_global_id(1);
    if (col >= N || row >= N) return;

    // z_0 = pixel coordinate; c is fixed.
    float zr = x_min + col * (x_max - x_min) / (float)N;
    float zi = y_min + row * (y_max - y_min) / (float)N;

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

# Burning Ship — take abs of z before each square. Note the view is upside
# down from the classic ship; we flip at render.
KERNELS["Burning Ship"] = """
__kernel void burning_ship(
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
        float ar = fabs(zr);
        float ai = fabs(zi);
        float tmp = ar*ar - ai*ai + c_real;
        zi = 2.0f * ar * ai + c_imag;
        zr = tmp;
        count++;
    }
    result[row * N + col] = count;
}
"""

# Tricorn — conj(z)^2 = (zr - i zi)^2 = zr^2 - zi^2 - 2 i zr zi.
# Only change vs Mandelbrot: the imaginary-update term is negated.
KERNELS["Tricorn"] = """
__kernel void tricorn(
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
        zi = -2.0f * zr * zi + c_imag;      // <-- sign flip
        zr = tmp;
        count++;
    }
    result[row * N + col] = count;
}
"""


def run_variant(ctx, queue, name, N, max_iter=MAX_ITER):
    """Compile and run one variant. Returns (image, elapsed_seconds)."""
    x_min, x_max, y_min, y_max = VIEWS[name]
    prog = cl.Program(ctx, KERNELS[name]).build()
    kernel_name = {
        "Mandelbrot":   "mandelbrot",
        "Julia":        "julia",
        "Burning Ship": "burning_ship",
        "Tricorn":      "tricorn",
    }[name]
    kernel = getattr(prog, kernel_name)

    result_host = np.zeros((N, N), dtype=np.int32)
    result_dev  = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, result_host.nbytes)

    # Julia takes two extra scalar args (c_real, c_imag).
    t0 = time.perf_counter()
    if name == "Julia":
        kernel(
            queue, (N, N), None, result_dev,
            np.float32(x_min), np.float32(x_max),
            np.float32(y_min), np.float32(y_max),
            np.float32(JULIA_C[0]), np.float32(JULIA_C[1]),
            np.int32(N), np.int32(max_iter),
        )
    else:
        kernel(
            queue, (N, N), None, result_dev,
            np.float32(x_min), np.float32(x_max),
            np.float32(y_min), np.float32(y_max),
            np.int32(N), np.int32(max_iter),
        )
    queue.finish()
    elapsed = time.perf_counter() - t0

    cl.enqueue_copy(queue, result_host, result_dev)
    queue.finish()
    return result_host, elapsed


if __name__ == "__main__":
    N = 1024

    ctx   = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    print(f"Device: {ctx.devices[0].name}\n")

    # Warm up each kernel (first run triggers OpenCL C compile).
    for name in KERNELS:
        run_variant(ctx, queue, name, N=64)

    images = {}
    for name in KERNELS:
        img, t = run_variant(ctx, queue, name, N)
        images[name] = img
        print(f"  {name:<13}: {t*1e3:6.1f} ms  (view={VIEWS[name]})")

    # 2x2 panel
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, (name, img) in zip(axes.flat, images.items()):
        # Burning Ship looks right-way-up when we flip imag axis at render.
        if name == "Burning Ship":
            img = img[::-1]
        ax.imshow(img, cmap='hot', origin='lower')
        ax.set_title(name, fontsize=12)
        ax.axis('off')

    plt.suptitle("Escape-time fractals (OpenCL)", fontsize=14)
    plt.tight_layout()
    out = Path(__file__).parent / "mandelbrot_variants.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out}")
