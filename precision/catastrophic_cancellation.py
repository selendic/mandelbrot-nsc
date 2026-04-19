import numpy as np


def quadratic_naive(a, b, c):
    """Compute quadratic roots directly using the standard formula."""
    t = type(a)
    # np.float32 or np.float64
    disc = t(np.sqrt(b * b - t(4) * a * c))  # b*b not b**2; t() casts literals and sqrt
    x1 = (-b + disc) / (t(2) * a)
    x2 = (-b - disc) / (t(2) * a)
    return x1, x2


# x^2 - 10000.0001*x + 1 = 0
# roots: x1 ~ 10000, x2 ~ 1e-4

def run_naive():
    """Print naive-root results for float32 and float64 precision."""
    for dtype in [np.float32, np.float64]:
        a, b, c = dtype(1.0), dtype(-10000.0001), dtype(1.0)
        x1, x2 = quadratic_naive(a, b, c)
        print(f"{dtype.__name__}: x1 = {float(x1):.4f}, x2 = {float(x2):.10f}")

    """
    float32: x1 = 10000.0000, x2 = 0.0000000000
    float64: x1 = 10000.0000, x2 = 0.0001000000
    """


def quadratic_stable(a, b, c):
    """Compute quadratic roots using a cancellation-avoiding stable formulation."""
    t = type(a)
    disc = t(np.sqrt(b * b - t(4) * a * c))
    if b > 0:
        x1 = (-b - disc) / (t(2) * a)  # pick sign that avoids cancellation
    else:
        x1 = (-b + disc) / (t(2) * a)
    x2 = c / (a * x1)  # Vieta’s formula: x1 * x2 = c/a
    return x1, x2


def run_both():
    """Compare relative error of naive and stable quadratic solvers."""
    true_small = 1.0 / 10000.0001  # ~ 1e-4
    for dtype in [np.float32, np.float64]:
        a, b, c = dtype(1.0), dtype(-10000.0001), dtype(1.0)
        _, x2_naive = quadratic_naive(a, b, c)
        _, x2_stable = quadratic_stable(a, b, c)
        err_naive = abs(float(x2_naive) - true_small) / true_small
        err_stable = abs(float(x2_stable) - true_small) / true_small
        print(f"{dtype.__name__}: err_naive={err_naive:.2e} err_stable={err_stable:.2e}")

    """
    float32: err_naive=1.00e+00 err_stable=1.53e-08
    float64: err_naive=1.20e-08 err_stable=1.00e-08
    """


if __name__ == "__main__":
    # run_naive()
    run_both()
