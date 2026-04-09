import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path

from mandelbrot_trajectory_divergence import escape_count


if __name__ == "__main__":
    N, MAX_ITER = 1024, 1000
    x_min, x_max = -0.750, -0.747
    y_min, y_max = 0.099, 0.101
    extent = (x_min, x_max, y_min, y_max)

    x = np.linspace(x_min, x_max, N)
    y = np.linspace(y_min, y_max, N)
    C = (x[np.newaxis, :] + 1j * y[:, np.newaxis]).astype(np.complex128)
    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "mandelbrot_sensitivity_map_condition_number.png"

    eps32 = float(np.finfo(np.float32).eps)
    delta = np.maximum(eps32 * np.abs(C), 1e-10)

    n_base = escape_count(C, MAX_ITER).astype(float)
    n_perturb = escape_count(C + delta, MAX_ITER).astype(float)

    dn = np.abs(n_base - n_perturb)
    kappa = np.where(n_base > 0, dn / (eps32 * n_base), np.nan)

    cmap_k = plt.cm.hot.copy()
    cmap_k.set_bad("0.25")
    vmax = np.nanpercentile(kappa, 99)

    fig, ax = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)

    im = ax.imshow(
        kappa, cmap=cmap_k, origin="lower", extent=extent,
        norm=LogNorm(vmin=1, vmax=vmax),
    )
    fig.colorbar(im, ax=ax, label=r"$\kappa(c)$  (log scale,  $\kappa \geq 1$)")
    ax.set_title(
        r"Condition number  $\kappa(c) = |\Delta n|\,/\,(\varepsilon_{32}\,n(c))$"
    )
    ax.set_xlabel("Re(c)")
    ax.set_ylabel("Im(c)")

    fig.savefig(out_path, dpi=300)

    plt.show()
