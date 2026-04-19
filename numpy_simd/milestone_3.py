import numpy as np
import time


def measure_row_vs_column_major_access(A):
    """Measure row-wise and column-wise traversal time for a square 2D array."""
    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]
    N = A.shape[0]

    t = time.perf_counter()
    for i in range(N): s = np.sum(A[i, :])
    t_row_major = time.perf_counter() - t

    t = time.perf_counter()
    for j in range(N): s = np.sum(A[:, j])
    t_column_major = time.perf_counter() - t

    return t_row_major, t_column_major


def main():
    """Compare access patterns for C-order and Fortran-order arrays."""
    N = 10000
    A = np.random.rand(N, N)

    t_row_major_normal, t_column_major_normal = measure_row_vs_column_major_access(A)

    f = np.asfortranarray(A)

    t_row_major_fortran, t_column_major_fortran = measure_row_vs_column_major_access(f)

    print(f"Row-major access (C order): {t_row_major_normal:.4f} s")
    print(f"Column-major access (C order): {t_column_major_normal:.4f} s")
    print(f"Row-major access (Fortran order): {t_row_major_fortran:.4f} s")
    print(f"Column-major access (Fortran order): {t_column_major_fortran:.4f} s")


if __name__ == "__main__":
    main()

    # Row-major access (C order): 0.0535 s
    # Column-major access (C order): 0.1336 s
    # Row-major access (Fortran order): 0.1424 s
    # Column-major access (Fortran order): 0.0518 s
