import dask, random, time, statistics
from dask import delayed
from dask.distributed import LocalCluster, Client


def monte_carlo_chunk(n_samples):
    """Return the number of in-circle hits for a Monte Carlo chunk."""
    inside = 0
    for _ in range(n_samples):
        x, y = random.random(), random.random()
        if x * x + y * y <= 1:
            inside += 1
    return inside


def serial(total, n_chunks, samples):
    """Run the chunked Monte Carlo baseline serially and report timing."""
    # Serial baseline
    t0 = time.perf_counter()
    results = [monte_carlo_chunk(samples) for _ in range(n_chunks)]
    t_serial = time.perf_counter() - t0
    print(f"Serial: {t_serial:.3f} s pi={4 * sum(results) / total:.4f}")
    return results, t_serial


def dask_delayed(total, n_chunks, samples):
    """Run the same chunked Monte Carlo workload using Dask delayed tasks."""
    # Dask delayed -- task graph is built, not executed yet
    tasks = [delayed(monte_carlo_chunk)(samples) for _ in range(n_chunks)]
    t0 = time.perf_counter()
    results = dask.compute(*tasks)
    t_dask = time.perf_counter() - t0
    print(f"Dask: {t_dask:.3f} s pi={4 * sum(results) / total:.4f}")

    # Visualise (requires: conda install python-graphviz)
    dask.visualize(*tasks, filename="task_graph.png")

    return results, t_dask


def locally_clustered(
        n_chunks: int,
        samples: int,
        scale: int = 1,
        n_workers: int | None = None,
        cluster=None,
        client=None,
):
    """Execute delayed Monte Carlo tasks on a local Dask cluster setup."""
    if cluster is None and n_workers is None:
        raise ValueError("Provide either an existing cluster or n_workers.")

    if scale < 1:
        print("scale must be >= 1; defaulting to 1.")
        scale = 1

    if cluster is None:
        # Create local cluster; start with max workers -- scale() adjusts without restarting
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)

    if client is None:
        client = Client(cluster)
        print(f"Dashboard: {client.dashboard_link}")
        # --> open the printed URL in your browser
    if scale > 1:
        # Vary n_workers: scale() resizes without restarting the scheduler
        # (recreating LocalCluster while the browser is open breaks the dashboard)
        cluster.scale(scale)
        client.wait_for_workers(scale)

    # Rerun E1 tasks; LocalCluster scheduler takes over
    tasks = [delayed(monte_carlo_chunk)(samples) for _ in range(n_chunks)]
    t0 = time.perf_counter()
    results = dask.compute(*tasks)
    t = time.perf_counter() - t0

    return results, t, cluster, client


def e1():
    """Run exercise 1: serial vs local Dask-delayed comparisons over chunk counts."""
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


def e2():
    """Run exercise 2: evaluate speedup across worker and chunk configurations."""
    total, n_chunks = 10_000_000, 8
    samples = total // n_chunks

    print(f"Running serial version for n_chunks={n_chunks}...")
    _, t_serial = serial(total, n_chunks, samples)

    print(f"Running locally clustered dask version for n_chunks={n_chunks} (and 4 workers)...")
    _, t_dask, cluster, client = locally_clustered(n_chunks, samples, scale=4, n_workers=16)

    print(f"Speedup: {t_serial / t_dask:.2f}x\n")

    for n_workers in [2, 4, 8, 16]:
        for n_chunks in [4, 8, 16, 32]:
            samples = total // n_chunks
            print(f"Running locally clustered dask version with {n_workers} workers (scale)...")
            _, t_serial = serial(total, n_chunks, samples)
            _, t_dask, cluster, client = locally_clustered(n_chunks, samples, scale=n_workers, cluster=cluster,
                                                           client=client)
            print(f"Speedup with {n_workers} workers and {n_chunks} chunks: {t_serial / t_dask:.2f}x\n")

    client.close()
    cluster.close()

    """
    Running serial version for n_chunks=8...
    Serial: 0.741 s pi=3.1423
    Running locally clustered dask version for n_chunks=8 (and 4 workers)...
    Dashboard: http://127.0.0.1:8787/status
    Speedup: 1.70x
    
    Running locally clustered dask version with 2 workers (scale)...
    Serial: 0.807 s pi=3.1414
    Speedup with 2 workers and 4 chunks: 2.02x
    
    Running locally clustered dask version with 2 workers (scale)...
    Serial: 0.755 s pi=3.1426
    Speedup with 2 workers and 8 chunks: 1.91x
    
    Running locally clustered dask version with 2 workers (scale)...
    Serial: 0.769 s pi=3.1412
    Speedup with 2 workers and 16 chunks: 1.93x
    
    Running locally clustered dask version with 2 workers (scale)...
    Serial: 0.767 s pi=3.1412
    Speedup with 2 workers and 32 chunks: 1.90x
    
    Running locally clustered dask version with 4 workers (scale)...
    Serial: 0.764 s pi=3.1417
    Speedup with 4 workers and 4 chunks: 3.49x
    
    Running locally clustered dask version with 4 workers (scale)...
    Serial: 0.764 s pi=3.1418
    Speedup with 4 workers and 8 chunks: 3.70x
    
    Running locally clustered dask version with 4 workers (scale)...
    Serial: 0.756 s pi=3.1417
    Speedup with 4 workers and 16 chunks: 3.30x
    
    Running locally clustered dask version with 4 workers (scale)...
    Serial: 0.753 s pi=3.1422
    Speedup with 4 workers and 32 chunks: 3.22x
    
    Running locally clustered dask version with 8 workers (scale)...
    Serial: 0.763 s pi=3.1418
    Speedup with 8 workers and 4 chunks: 3.40x
    
    Running locally clustered dask version with 8 workers (scale)...
    Serial: 0.769 s pi=3.1423
    Speedup with 8 workers and 8 chunks: 5.36x
    
    Running locally clustered dask version with 8 workers (scale)...
    Serial: 0.772 s pi=3.1421
    Speedup with 8 workers and 16 chunks: 5.17x
    
    Running locally clustered dask version with 8 workers (scale)...
    Serial: 0.773 s pi=3.1417
    Speedup with 8 workers and 32 chunks: 5.04x
    
    Running locally clustered dask version with 16 workers (scale)...
    Serial: 0.747 s pi=3.1418
    Speedup with 16 workers and 4 chunks: 3.07x
    
    Running locally clustered dask version with 16 workers (scale)...
    Serial: 0.830 s pi=3.1411
    Speedup with 16 workers and 8 chunks: 4.79x
    
    Running locally clustered dask version with 16 workers (scale)...
    Serial: 0.765 s pi=3.1411
    Speedup with 16 workers and 16 chunks: 5.62x
    
    Running locally clustered dask version with 16 workers (scale)...
    Serial: 0.761 s pi=3.1413
    Speedup with 16 workers and 32 chunks: 5.27x
    """


if __name__ == "__main__":
    # e1()
    e2()
