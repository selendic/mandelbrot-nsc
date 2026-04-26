#pragma OPENCL EXTENSION cl_khr_fp64 : enable

import matplotlib.pyplot as plt
import numpy as np
import pyopencl as cl
import time


from numba_jit import mandelbrot_numba_jit
from gpu_util import generate_complex_grid

KERNEL_SRC = """
__kernel void mandelbrot(
    __global int *result,
    const double x_min, const double x_max,
    const double y_min, const double y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (col >= N || row >= N) return;   // guard against over-launch

    double c_real = x_min + col * (x_max - x_min) / (double) (N - 1);
    double c_imag = y_min + row * (y_max - y_min) / (double) (N - 1);

    double zr = 0.0, zi = 0.0;
    int count = max_iter;

    for (int n = 0; n < max_iter; ++n) {
        double tmp = zr*zr - zi*zi + c_real;
        zi = 2.0 * zr * zi + c_imag;
        zr = tmp;

        if (zr*zr + zi*zi > 4.0) {
            count = n;   // match Numba's returned n
            break;
        }
    }
    result[row * N + col] = count;
}
"""

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
prog = cl.Program(ctx, KERNEL_SRC).build()

N, MAX_ITER = 1024, 200
X_MIN, X_MAX = -2.0, 1.0
Y_MIN, Y_MAX = -1.5, 1.5

image = np.zeros((N, N), dtype=np.int32)
image_dev = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, image.nbytes)

# Retrieve the kernel function from the compiled program
mandelbrot_kernel = cl.Kernel(prog, "mandelbrot")

# --- Warm up (first launch triggers a kernel compile) ---
mandelbrot_kernel(queue, (64, 64), None, image_dev,
                  np.float64(X_MIN), np.float64(X_MAX),
                  np.float64(Y_MIN), np.float64(Y_MAX),
                  np.int32(64), np.int32(MAX_ITER))
queue.finish()

# --- Time the real run ---
t0 = time.perf_counter()
mandelbrot_kernel(
    queue, (N, N), None,  # global size (N, N); let OpenCL pick local
    image_dev,
    np.float64(X_MIN), np.float64(X_MAX),
    np.float64(Y_MIN), np.float64(Y_MAX),
    np.int32(N), np.int32(MAX_ITER),
)
queue.finish()
elapsed = time.perf_counter() - t0

cl.enqueue_copy(queue, image, image_dev)
queue.finish()

print(f"GPU {N}x{N}: {elapsed * 1e3:.3f} ms")
plt.imshow(image, cmap='hot', origin='lower')
plt.axis('off')
plt.savefig("mandelbrot_gpu_float64.png", dpi=150, bbox_inches='tight')

# Compare with numba_jit version for sanity check
C = generate_complex_grid(N, dtype=np.complex128)
# Numba warmup:
mandelbrot_numba_jit.mandelbrot_naive_full_numba_parallel(C, threshold=2.0,
                                                                          max_iter=MAX_ITER,
                                                                          dtype_int=np.int32,
                                                                          dtype_complex=np.complex128)
t0 = time.perf_counter()
numba_jit_res = mandelbrot_numba_jit.mandelbrot_naive_full_numba_parallel(C, threshold=2.0,
                                                                          max_iter=MAX_ITER,
                                                                          dtype_int=np.int32,
                                                                          dtype_complex=np.complex128)
elapsed_cpu = time.perf_counter() - t0
print(f"CPU {N}x{N}: {elapsed_cpu * 1e3:.3f} ms")
plt.imshow(numba_jit_res, cmap='hot', origin='lower')
plt.axis('off')
plt.savefig("mandelbrot_cpu_float64.png", dpi=150, bbox_inches='tight')
print(f"GPU res == Numba res?: {np.array_equal(numba_jit_res, image)}")
print(f"Number of mismatches: {np.count_nonzero(numba_jit_res != image)}")
# Pinpoint the mismatches
mismatches = np.zeros_like(numba_jit_res)
mismatches[numba_jit_res != image] = 1
plt.imshow(mismatches, cmap='gray', origin='lower')
plt.axis('off')
plt.savefig("mandelbrot_mismatches.png", dpi=150, bbox_inches="tight")

"""
GPU 1024x1024: 4.033 ms
CPU 1024x1024: 15.329 ms
GPU res == Numba res?: True
Number of mismatches: 0
"""
