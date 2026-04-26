import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

"""
In this specific script, but also during the course up until now,
AI helped in creating the charts, so that they look nice and clear.
I concluded that that particular use-case is similar to generating test cases,
if not even more leaning to the side of encouraged AI use,
since this is not a course about data visualization or design after all. :)
"""

data = {
    "1024×1024": [
        ("Naive", 2.3765),
        ("Numpy", 0.3883),
        ("Numba Hybrid", 0.2645),
        ("Numba Full", 0.0419),
        ("Numba Parallel", 0.0100),
        ("Multiprocessing", 0.0109),
        ("Dask local", 0.0417),
        ("Dask cluster", 0.0813),
        ("GPU fp32", 0.0009),
        ("GPU fp64", 0.0050),
    ],
    "4096×4096": [
        ("Naive", 38.1425),
        ("Numpy", 24.4803),
        ("Numba Hybrid", 4.2149),
        ("Numba Full", 0.6738),
        ("Numba Parallel", 0.1321),
        ("Multiprocessing", 0.0989),
        ("Dask local", 0.3326),
        ("Dask cluster", 0.4801),
        ("GPU fp32", 0.0597),
        ("GPU fp64", 0.1250),
    ],
}

palette = {
    "Naive": "#dc2626",
    "Numpy": "#f97316",
    "Numba Hybrid": "#f59e0b",
    "Numba Full": "#d97706",
    "Numba Parallel": "#ca8a04",
    "Multiprocessing": "#15803d",
    "Dask local": "#0369a1",
    "Dask cluster": "#0891b2",
    "GPU fp32": "#7c3aed",
    "GPU fp64": "#6d28d9",
}


def fmt(v, _):
    if v < 0.001:
        return f"{v:.4f}"
    if v < 0.01:
        return f"{v:.3f}"
    if v < 1:
        return f"{v:.2f}"
    return f"{v:.0f}"


def output_filename(size_label):
    return f"benchmark_bars_{size_label.replace('×', 'x')}.png"


for sz, rows in data.items():
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("white")

    methods = [r[0] for r in rows]
    times = [r[1] for r in rows]
    colors = [palette[m] for m in methods]
    y_pos = np.arange(len(methods))

    bars = ax.barh(y_pos, times, color=colors, edgecolor="none", height=0.65)

    ax.set_xscale("log")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_title(sz, fontsize=14, fontweight="bold", pad=10)
    ax.set_facecolor("#f9f9f9")
    ax.grid(axis="x", which="both", linestyle="--", linewidth=0.5,
            color="white", zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", length=0)

    ax.set_xlim(min(times) * 0.4, max(times) * 5)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt))

    for bar, t in zip(bars, times):
        ax.text(
            t * 1.2, bar.get_y() + bar.get_height() / 2,
            f"{t:.4f}s", va="center", ha="left", fontsize=8.5, color="#333"
        )

    fig.suptitle(
        "Benchmark Runtimes (log scale) — lower is better",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    filename = output_filename(sz)
    fig.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {filename}")
