import numpy as np


def find_machine_epsilon(dtype=np.float64):
    """Return machine epsilon for the given floating-point dtype."""
    eps = dtype(1.0)
    while dtype(1.0) + eps / dtype(2.0) != dtype(1.0):
        eps = eps / dtype(2.0)
    return eps


if __name__ == "__main__":
    for dtype in [np.float16, np.float32, np.float64]:
        computed = find_machine_epsilon(dtype)
        reference = np.finfo(dtype).eps
        print(f"{dtype.__name__}:")
        print(f"\tComputed: {float(computed):.4e}")
        print(f"\tnp.finfo: {float(reference):.4e}")

    """
    float16:
        Computed: 9.7656e-04
        np.finfo: 9.7656e-04
    float32:
        Computed: 1.1921e-07
        np.finfo: 1.1921e-07
    float64:
        Computed: 2.2204e-16
        np.finfo: 2.2204e-16
    """
