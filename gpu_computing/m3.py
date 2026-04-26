#!/usr/bin/env python3
"""Benchmark Mandelbrot on CPU (Numba) vs GPU (PyOpenCL) for float32/float64."""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import numpy as np
import pyopencl as cl

from numba_jit import mandelbrot_numba_jit
from gpu_util import generate_complex_grid

DEFAULT_RESOLUTION = 1024
MAX_ITER = 200
RUNS = 5
WARMUP_RUNS = 1

X_MIN, X_MAX = -2.0, 1.0
Y_MIN, Y_MAX = -1.5, 1.5

KERNEL_FLOAT32 = """
__kernel void mandelbrot(
    __global int *result,
    const float x_min, const float x_max,
    const float y_min, const float y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (col >= N || row >= N) return;

    float c_real = x_min + col * (x_max - x_min) / (float)(N - 1);
    float c_imag = y_min + row * (y_max - y_min) / (float)(N - 1);

    float zr = 0.0f, zi = 0.0f;
    int count = max_iter;

    for (int n = 0; n < max_iter; ++n) {
        float tmp = zr * zr - zi * zi + c_real;
        zi = 2.0f * zr * zi + c_imag;
        zr = tmp;

        if (zr * zr + zi * zi > 4.0f) {
            count = n;
            break;
        }
    }
    result[row * N + col] = count;
}
"""

KERNEL_FLOAT64 = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void mandelbrot(
    __global int *result,
    const double x_min, const double x_max,
    const double y_min, const double y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (col >= N || row >= N) return;

    double c_real = x_min + col * (x_max - x_min) / (double)(N - 1);
    double c_imag = y_min + row * (y_max - y_min) / (double)(N - 1);

    double zr = 0.0, zi = 0.0;
    int count = max_iter;

    for (int n = 0; n < max_iter; ++n) {
        double tmp = zr * zr - zi * zi + c_real;
        zi = 2.0 * zr * zi + c_imag;
        zr = tmp;

        if (zr * zr + zi * zi > 4.0) {
            count = n;
            break;
        }
    }
    result[row * N + col] = count;
}
"""


@dataclass
class BenchResult:
    times_ms: list[float]
    output: np.ndarray

    @property
    def median_ms(self) -> float:
        return float(statistics.median(self.times_ms))

    @property
    def mean_ms(self) -> float:
        return float(statistics.mean(self.times_ms))

    @property
    def std_ms(self) -> float:
        # Match numpy's default population std for simple reporting.
        return float(np.std(self.times_ms))


def benchmark(name: str, run_once, warmups: int = WARMUP_RUNS, runs: int = RUNS) -> BenchResult:
    for _ in range(warmups):
        run_once()

    times_ms: list[float] = []
    output: np.ndarray | None = None
    for _ in range(runs):
        t0 = time.perf_counter()
        output = run_once()
        dt_ms = (time.perf_counter() - t0) * 1e3
        times_ms.append(dt_ms)

    if output is None:
        raise RuntimeError(f"No output produced for benchmark {name}.")

    print(f"{name:<12} mean+-std: {statistics.mean(times_ms):8.3f} +- {np.std(times_ms):7.3f} ms")
    return BenchResult(times_ms=times_ms, output=output)


def make_cpu_runner(n: int, max_iter: int, dtype_complex):
    c_grid = generate_complex_grid(n, dtype=dtype_complex)
    threshold = np.float32(2.0) if dtype_complex == np.complex64 else 2.0

    def run_once() -> np.ndarray:
        return mandelbrot_numba_jit.mandelbrot_naive_full_numba_parallel(
            c_grid,
            threshold=threshold,
            max_iter=max_iter,
            dtype_int=np.int32,
            dtype_complex=dtype_complex,
        )

    return run_once


def make_gpu_runner(n: int, max_iter: int, is_float64: bool):
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    if is_float64:
        has_fp64 = any("cl_khr_fp64" in dev.extensions for dev in ctx.devices)
        if not has_fp64:
            raise RuntimeError("Selected OpenCL device does not report cl_khr_fp64 support.")

    kernel_src = KERNEL_FLOAT64 if is_float64 else KERNEL_FLOAT32
    program = cl.Program(ctx, kernel_src).build()
    kernel = cl.Kernel(program, "mandelbrot")

    image_host = np.empty((n, n), dtype=np.int32)
    image_dev = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, image_host.nbytes)

    scalar = np.float64 if is_float64 else np.float32
    x_min = scalar(X_MIN)
    x_max = scalar(X_MAX)
    y_min = scalar(Y_MIN)
    y_max = scalar(Y_MAX)
    n_i32 = np.int32(n)
    max_i32 = np.int32(max_iter)

    def run_once() -> np.ndarray:
        kernel(
            queue,
            (n, n),
            None,
            image_dev,
            x_min,
            x_max,
            y_min,
            y_max,
            n_i32,
            max_i32,
        )
        cl.enqueue_copy(queue, image_host, image_dev)
        queue.finish()
        return image_host.copy()

    return run_once


def print_median_table(results: dict[tuple[str, str], BenchResult]) -> None:
    header = f"{'Precision':<10} | {'CPU median (ms)':>15} | {'GPU median (ms)':>15}"
    print("\nMedian runtime table")
    print(header)
    print("-" * len(header))
    for precision in ("float32", "float64"):
        cpu = results[(precision, "CPU")].median_ms
        gpu = results[(precision, "GPU")].median_ms
        print(f"{precision:<10} | {cpu:15.3f} | {gpu:15.3f}")


def main(n: int = DEFAULT_RESOLUTION, max_iter: int = MAX_ITER) -> None:
    results: dict[tuple[str, str], BenchResult] = {}

    print(f"Benchmark setup: N={n}, max_iter={max_iter}, warmup={WARMUP_RUNS}, runs={RUNS}\n")

    cpu32 = benchmark("CPU float32", make_cpu_runner(n, max_iter, np.complex64))
    gpu32 = benchmark("GPU float32", make_gpu_runner(n, max_iter, is_float64=False))
    cpu64 = benchmark("CPU float64", make_cpu_runner(n, max_iter, np.complex128))
    gpu64 = benchmark("GPU float64", make_gpu_runner(n, max_iter, is_float64=True))

    results[("float32", "CPU")] = cpu32
    results[("float32", "GPU")] = gpu32
    results[("float64", "CPU")] = cpu64
    results[("float64", "GPU")] = gpu64

    print_median_table(results)

    eq32 = np.array_equal(cpu32.output, gpu32.output)
    eq64 = np.array_equal(cpu64.output, gpu64.output)
    mism32 = int(np.count_nonzero(cpu32.output != gpu32.output))
    mism64 = int(np.count_nonzero(cpu64.output != gpu64.output))

    print("\nEquality checks (CPU vs GPU)")
    print(f"float32 assert (array_equal): {eq32}  mismatches: {mism32} (={mism32 / (n * n) * 100:.2f}%)")
    print(f"float64 assert (array_equal): {eq64}  mismatches: {mism64} (={mism64 / (n * n) * 100:.2f}%)")


if __name__ == "__main__":
    for ns in [1024, 2048, 4096, 8192]:
        print(f"\n{'=' * 40}\nRunning benchmarks for N={ns}\n{'=' * 40}")
        main(n=ns, max_iter=200)


"""
========================================
Running benchmarks for N=1024
========================================
Benchmark setup: N=1024, max_iter=200, warmup=1, runs=5

CPU float32  mean+-std:   82.578 +-   0.605 ms
GPU float32  mean+-std:    1.111 +-   0.321 ms
CPU float64  mean+-std:   82.276 +-   0.025 ms
GPU float64  mean+-std:    4.986 +-   0.282 ms

Median runtime table
Precision  | CPU median (ms) | GPU median (ms)
----------------------------------------------
float32    |          82.268 |           0.975
float64    |          82.276 |           4.862

Equality checks (CPU vs GPU)
float32 assert (array_equal): False  mismatches: 1123 (=0.11%)
float64 assert (array_equal): True  mismatches: 0 (=0.00%)

========================================
Running benchmarks for N=2048
========================================
Benchmark setup: N=2048, max_iter=200, warmup=1, runs=5

CPU float32  mean+-std:  329.228 +-   2.096 ms
GPU float32  mean+-std:    5.339 +-   0.929 ms
CPU float64  mean+-std:  329.442 +-   2.160 ms
GPU float64  mean+-std:   20.596 +-   1.131 ms

Median runtime table
Precision  | CPU median (ms) | GPU median (ms)
----------------------------------------------
float32    |         328.197 |           4.950
float64    |         328.354 |          20.018

Equality checks (CPU vs GPU)
float32 assert (array_equal): False  mismatches: 4290 (=0.10%)
float64 assert (array_equal): False  mismatches: 2 (=0.00%)

========================================
Running benchmarks for N=4096
========================================
Benchmark setup: N=4096, max_iter=200, warmup=1, runs=5

CPU float32  mean+-std: 1339.136 +-   7.844 ms
GPU float32  mean+-std:   16.304 +-   0.412 ms
CPU float64  mean+-std: 1336.156 +-   1.874 ms
GPU float64  mean+-std:   74.248 +-   0.730 ms

Median runtime table
Precision  | CPU median (ms) | GPU median (ms)
----------------------------------------------
float32    |        1336.070 |          16.298
float64    |        1336.046 |          74.160

Equality checks (CPU vs GPU)
float32 assert (array_equal): False  mismatches: 35375 (=0.21%)
float64 assert (array_equal): False  mismatches: 2 (=0.00%)

========================================
Running benchmarks for N=8192
========================================
Benchmark setup: N=8192, max_iter=200, warmup=1, runs=5

CPU float32  mean+-std: 5341.589 +-  15.763 ms
GPU float32  mean+-std:   62.371 +-   0.303 ms
CPU float64  mean+-std: 5337.793 +-   3.794 ms
GPU float64  mean+-std:  289.376 +-   0.671 ms

Median runtime table
Precision  | CPU median (ms) | GPU median (ms)
----------------------------------------------
float32    |        5346.941 |          62.442
float64    |        5338.171 |         289.692

Equality checks (CPU vs GPU)
float32 assert (array_equal): False  mismatches: 96812 (=0.14%)
float64 assert (array_equal): False  mismatches: 15 (=0.00%)
"""
