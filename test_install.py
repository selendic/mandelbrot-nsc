import numpy as np
import matplotlib
import numba as nb
import dask, distributed


def print_all_versions():
    print(f"NumPy version: {np.__version__}")
    print(f"Matplotlib version: {matplotlib.__version__}")
    print(f"Numba version: {nb.__version__}")
    print(f"Dask version: {dask.__version__}")
    print(f"Distributed version: {distributed.__version__}")


def dask_sanity_check():
    from dask.distributed import LocalCluster, Client
    cluster = LocalCluster(n_workers=2, threads_per_worker=1)
    client = Client(cluster)
    result = dask.delayed(sum)(range(1000))
    print(dask.compute(result))
    client.close()
    cluster.close()


if __name__ == "__main__":
    # print_all_versions()
    dask_sanity_check()
