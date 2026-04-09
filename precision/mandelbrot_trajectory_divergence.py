import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def escape_count(C, max_iter):
    z = np.zeros_like(C)
    cnt = np.full(C.shape, max_iter, dtype=np.int32)
    esc = np.zeros(C.shape, dtype=bool)
    for k in range(max_iter):
        z[~esc] = z[~esc] ** 2 + C[~esc]
        newly = ~esc & (np.abs(z) > 2.0)
        cnt[newly] = k
        esc[newly] = True
    return cnt


if __name__ == "__main__":
    N, MAX_ITER = 1024, 1000
    TAUS = [0.01, 0.001]
    x_min, x_max = -0.750, -0.747
    y_min, y_max = 0.099, 0.101
    x = np.linspace(x_min, x_max, N)
    y = np.linspace(y_min, y_max, N)
    extent = (x_min, x_max, y_min, y_max)

    C64 = (x[np.newaxis, :] + 1j * y[:, np.newaxis]).astype(np.complex128)
    C32 = C64.astype(np.complex64)

    escape64 = escape_count(C64, MAX_ITER)
    escape32 = escape_count(C32, MAX_ITER)

    out_dir = Path(__file__).resolve().parent
    escape64_path = out_dir / "mandelbrot_escape_complex128.png"
    escape32_path = out_dir / "mandelbrot_escape_complex64.png"

    fig64, ax64 = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im64 = ax64.imshow(escape64, cmap="magma", origin="lower", extent=extent, vmin=0, vmax=MAX_ITER)
    ax64.set_title("Mandelbrot escape (complex128)")
    ax64.set_xlabel("Re(c)")
    ax64.set_ylabel("Im(c)")
    fig64.colorbar(im64, ax=ax64, label="Escape iteration")
    fig64.savefig(escape64_path, dpi=300)
    plt.close(fig64)

    fig32, ax32 = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im32 = ax32.imshow(escape32, cmap="magma", origin="lower", extent=extent, vmin=0, vmax=MAX_ITER)
    ax32.set_title("Mandelbrot escape (complex64)")
    ax32.set_xlabel("Re(c)")
    ax32.set_ylabel("Im(c)")
    fig32.colorbar(im32, ax=ax32, label="Escape iteration")
    fig32.savefig(escape32_path, dpi=300)
    plt.close(fig32)

    for TAU in TAUS:
        mantissa, exponent = f"{TAU:.6e}".split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        tau_token = f"{mantissa}e{int(exponent):+03d}"
        divergence_path = out_dir / f"mandelbrot_trajectory_divergence_tau_{tau_token}.png"

        z32 = np.zeros_like(C32)
        z64 = np.zeros_like(C64)
        diverge = np.full((N, N), MAX_ITER, dtype=np.int32)
        active = np.ones((N, N), dtype=bool)
        for k in range(MAX_ITER):
            if not active.any(): break
            z32[active] = z32[active] ** 2 + C32[active]
            z64[active] = z64[active] ** 2 + C64[active]
            diff = np.abs(z32.astype(np.complex128) - z64)
            newly = active & (diff > TAU)
            diverge[newly] = k
            active[newly] = False

            fig_div, ax_div = plt.subplots(figsize=(6, 5), constrained_layout=True)
            im_div = ax_div.imshow(diverge, cmap="plasma", origin="lower", extent=extent, vmin=0, vmax=MAX_ITER)
            ax_div.set_title(f"Trajectory divergence (tau={TAU})")
            ax_div.set_xlabel("Re(c)")
            ax_div.set_ylabel("Im(c)")
            fig_div.colorbar(im_div, ax=ax_div, label="First divergence iteration")
            fig_div.savefig(divergence_path, dpi=300)
            plt.close(fig_div)

    """
    What fraction of pixels diverge before max_iter?
    The third panel is overwhelmingly dark purple (= low divergence iteration, i.e. diverged early), with only a small bright cluster around the main bulge/interior. The vast majority of pixels — likely >90% — diverge before MAX_ITER. The ones that don't (bright yellow, ~1000) are confined to the clearly bounded interior region.
    
    Where do trajectories diverge early?
    Early divergence (dark, low values in panel 3) corresponds precisely to the exterior black regions in panels 1 & 2 — points that escape quickly. The spiral filaments and boundary detail that glow brightly in the escape maps also appear somewhat elevated in panel 3, but most of the image floor is dark.
    
    Does early divergence correlate with high escape iteration counts?
    No — it's the inverse. Low divergence iteration tracks low escape iteration (exterior), and high divergence iteration tracks high escape iteration (interior/boundary). They are positively correlated structurally, as we established.
    """
