import numpy as np


def generate_complex_grid(image_size: int, dtype=np.complex128) -> np.ndarray:
    if dtype == np.complex128:
        f = np.float64
    elif dtype == np.complex64:
        f = np.float32
    else:
        raise ValueError("Unsupported dtype for complex grid. Use np.complex128 or np.complex64.")

    cols = np.arange(image_size, dtype=f(0).dtype)
    rows = np.arange(image_size, dtype=f(0).dtype)

    # Match kernel exactly: x_min + col * (x_max - x_min) / (N - 1)
    xs = f(-2.0) + (cols * (f(1.0)  - f(-2.0))) / f(image_size - 1)
    ys = f(-1.5) + (rows * (f(1.5)  - f(-1.5))) / f(image_size - 1)

    X, Y = np.meshgrid(xs, ys)

    C = np.empty((image_size, image_size), dtype=dtype)
    C.real[:] = X
    C.imag[:] = Y
    return C
