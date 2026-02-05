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

if __name__ == "__main__":
    print_all_versions()
