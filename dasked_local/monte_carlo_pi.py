import dask, random, time, statistics
from dask import delayed


def monte_carlo_chunk(n_samples):
    inside = 0
    for _ in range(n_samples):
        x, y = random.random(), random.random()
        if x * x + y * y <= 1:
            inside += 1
    return inside


def serial(total, n_chunks, samples):
    # Serial baseline
    t0 = time.perf_counter()
    results = [monte_carlo_chunk(samples) for _ in range(n_chunks)]
    t_serial = time.perf_counter() - t0
    print(f"Serial: {t_serial:.3f} s pi={4 * sum(results) / total:.4f}")
    return results, t_serial


def dask_delayed(total, n_chunks, samples):
    # Dask delayed -- task graph is built, not executed yet
    tasks = [delayed(monte_carlo_chunk)(samples) for _ in range(n_chunks)]
    t0 = time.perf_counter()
    results = dask.compute(*tasks)
    t_dask = time.perf_counter() - t0
    print(f"Dask: {t_dask:.3f} s pi={4 * sum(results) / total:.4f}")

    # Visualise (requires: conda install python-graphviz)
    dask.visualize(*tasks, filename="task_graph.png")

    return results, t_dask


if __name__ == "__main__":
    total, n_chunks = 1_000_000, 8
    samples = total // n_chunks

    print("Running serial version...")
    _, t_serial = serial(total, n_chunks, samples)

    print("Running dask version...")
    _, t_dask = dask_delayed(total, n_chunks, samples)

    print(f"\n Speedup: {t_serial / t_dask:.2f}x")

    for n_chunks in [4, 8, 16, 32]:
        samples = total // n_chunks
        print(f"Running dask version with {n_chunks} chunks...")
        _, t_serial = serial(total, n_chunks, samples)
        _, t_dask = dask_delayed(total, n_chunks, samples)
        print(f"Speedup with {n_chunks} chunks: {t_serial / t_dask:.2f}x\n")

"""
Running serial version...
Serial: 0.074 s pi=3.1404
Running dask version...
Dask: 0.184 s pi=3.1410

 Speedup: 0.40x
Running dask version with 4 chunks...
Serial: 0.072 s pi=3.1430
Dask: 0.080 s pi=3.1397
Speedup with 4 chunks: 0.90x

Running dask version with 8 chunks...
Serial: 0.076 s pi=3.1418
Dask: 0.081 s pi=3.1393
Speedup with 8 chunks: 0.94x

Running dask version with 16 chunks...
Serial: 0.074 s pi=3.1429
Dask: 0.081 s pi=3.1411
Speedup with 16 chunks: 0.92x

Running dask version with 32 chunks...
Serial: 0.074 s pi=3.1401
Dask: 0.081 s pi=3.1428
Speedup with 32 chunks: 0.92x
"""
