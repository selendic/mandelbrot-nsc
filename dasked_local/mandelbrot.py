from dask import delayed
from dask.distributed import Client, LocalCluster
import dask, numpy as np, time, statistics
from cpu_parallelization.mandelbrot_numba_parallel import mandelbrot_chunk, mandelbrot_serial


# mandelbrot_chunk: your @njit(cache=True) function from L04/L05
def mandelbrot_dask(N, x_min, x_max, y_min, y_max,
                    max_iter=100, n_chunks=32):
    chunk_size = max(1, N // n_chunks)
    tasks, row = [], 0
    while row < N:
        row_end = min(row + chunk_size, N)
        tasks.append(
            delayed(mandelbrot_chunk)(
                row, row_end, N, x_min, x_max, y_min, y_max, max_iter=max_iter
            )
        )
        row = row_end
    parts = dask.compute(*tasks)
    return np.vstack(parts)


if __name__ == "__main__":
    N, max_iter = 1024, 100
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25
    cluster = LocalCluster(n_workers=8, threads_per_worker=1)
    client = Client(cluster)
    # warm up all workers
    client.run(lambda: mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter=10))

    ref = mandelbrot_serial(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter=max_iter)
    result = mandelbrot_dask(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
    if not np.array_equal(ref, result):
        raise AssertionError("Dask result does not match serial reference output.")

    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        result = mandelbrot_dask(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
        times.append(time.perf_counter() - t0)
    print("Verification passed: dask output matches serial output.")
    print(f"Dask local (3 runs, n_chunks=32): median {statistics.median(times):.3f} s")
    client.close()
    cluster.close()

    """
    Verification passed: dask output matches serial output.
    Dask local (3 runs, n_chunks=32): median 0.046 s
    """
