from dask import delayed
from dask.distributed import Client, LocalCluster
import dask, numpy as np, time, statistics
import math
import matplotlib.pyplot as plt
from numba import njit

REMOTE_CLIENT_STR = "tcp://10.92.1.177:8786"


@njit(cache=True)
def mandelbrot_chunk_early_exit(row_start: int, row_end: int,
								col_start: int, col_end: int, N: int,
								x_min: float, x_max: float, y_min: float, y_max: float,
								threshold=2.0, max_iter=100, use_early_exit=True):
	"""Compute one square tile and stop early once all points in the tile have diverged."""
	height = row_end - row_start
	width = col_end - col_start
	result = np.full((height, width), max_iter, dtype=np.int32)
	zr = np.zeros((height, width), dtype=np.float64)
	zi = np.zeros((height, width), dtype=np.float64)
	active = np.ones((height, width), dtype=np.bool_)

	dx = (x_max - x_min) / N
	dy = (y_max - y_min) / N
	t2 = threshold * threshold
	active_count = height * width

	for it in range(max_iter):
		for r in range(height):
			y = y_min + (row_start + r) * dy
			for j in range(width):
				if active[r, j]:
					x = x_min + (col_start + j) * dx
					zr_old = zr[r, j]
					zi_old = zi[r, j]
					zr_new = zr_old * zr_old - zi_old * zi_old + x
					zi_new = 2.0 * zr_old * zi_old + y
					zr[r, j] = zr_new
					zi[r, j] = zi_new
					if zr_new * zr_new + zi_new * zi_new > t2:
						result[r, j] = it
						active[r, j] = False
						active_count -= 1

		if use_early_exit and active_count == 0:
			break

	return result


# mandelbrot_chunk: your @njit(cache=True) function from L04/L05
def mandelbrot_dask(N, x_min, x_max, y_min, y_max,
					max_iter=100, n_chunks=32, use_early_exit=True):
	# Convert total chunk target to a square tiling layout (chunks per axis).
	# We can safely assume that n_chunks is a perfect square.
	chunks_per_axis = max(1, math.isqrt(n_chunks))

	# Compute tile edge length from grid size and tiling layout.
	tile_size = max(1, math.ceil(N / chunks_per_axis))
	tasks = []
	tile_bounds = []
	for row_start in range(0, N, tile_size):
		row_end = min(row_start + tile_size, N)
		for col_start in range(0, N, tile_size):
			col_end = min(col_start + tile_size, N)
			# Store destination slice so we can stitch the computed tiles back in order.
			tile_bounds.append((row_start, row_end, col_start, col_end))
			# Create one delayed task per square tile.
			tasks.append(
				delayed(mandelbrot_chunk_early_exit)(
					row_start,
					row_end,
					col_start,
					col_end,
					N,
					x_min,
					x_max,
					y_min,
					y_max,
					max_iter=max_iter,
					use_early_exit=use_early_exit,
				)
			)

	# Trigger Dask execution for all tile tasks.
	tiles = dask.compute(*tasks)
	# Allocate final output image for tile assembly.
	result = np.empty((N, N), dtype=np.int32)
	for (row_start, row_end, col_start, col_end), tile in zip(tile_bounds, tiles):
		result[row_start:row_end, col_start:col_end] = tile
	return result


def mandelbrot_serial_blocks(N, x_min, x_max, y_min, y_max,
							 max_iter=100, n_chunks=32, use_early_exit=True):
	"""Serial baseline that uses the same square-tile decomposition as the Dask path."""
	# Use the same square tiling strategy as the parallel version.
	chunks_per_axis = max(1, math.isqrt(n_chunks))
	# Compute tile edge length for serial tile traversal.
	tile_size = max(1, math.ceil(N / chunks_per_axis))

	# Allocate final output and fill it tile-by-tile.
	result = np.empty((N, N), dtype=np.int32)
	for row_start in range(0, N, tile_size):
		row_end = min(row_start + tile_size, N)
		for col_start in range(0, N, tile_size):
			col_end = min(col_start + tile_size, N)
			tile = mandelbrot_chunk_early_exit(
				row_start,
				row_end,
				col_start,
				col_end,
				N,
				x_min,
				x_max,
				y_min,
				y_max,
				max_iter=max_iter,
				use_early_exit=use_early_exit,
			)
			result[row_start:row_end, col_start:col_end] = tile
	return result


def m1():
	N, max_iter = 1024, 100
	X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25
	# cluster = LocalCluster(n_workers=8, threads_per_worker=1)
	# client = Client(cluster) ->
	client = Client(REMOTE_CLIENT_STR)
	# warm up all workers
	client.run(lambda: mandelbrot_chunk_early_exit(0, 8, 0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter=10))

	# Reference result from one full-grid kernel call.
	ref = mandelbrot_chunk_early_exit(0, N, 0, N, N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter=max_iter)
	# Dask-computed result to compare against the reference.
	result = mandelbrot_dask(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
	if not np.array_equal(ref, result):
		raise AssertionError("Dask result does not match serial reference output.")

	times = []
	for _ in range(3):
		# Measure one full Dask run.
		t0 = time.perf_counter()
		result = mandelbrot_dask(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
		times.append(time.perf_counter() - t0)
	print("Verification passed: dask output matches serial output.")
	print(f"Dask local (3 runs, n_chunks=32): median {statistics.median(times):.3f} s")
	client.close()
	# cluster.close()

	"""
	Verification passed: dask output matches serial output.
	Dask local (3 runs, n_chunks=32): median 0.046 s
	"""


def _median_runtime(func, runs=3):
	times = []
	for _ in range(runs):
		# Time a single call and append elapsed seconds.
		t0 = time.perf_counter()
		func()
		times.append(time.perf_counter() - t0)
	return statistics.median(times)


def _run_m2_case(use_early_exit):
	workers = 16
	n_values = [1024, 2048, 4096, 8192]
	chunk_multipliers = [1, 2, 4, 8, 16, 32, 64, 128]
	max_iter = 100
	x_min, x_max, y_min, y_max = -2.5, 1.0, -1.25, 1.25
	n_runs = 5
	serial_baseline_chunks = 1
	mode = "with_early_exit" if use_early_exit else "without_early_exit"

	cluster = LocalCluster(n_workers=workers, threads_per_worker=1)
	client = Client(cluster)

	# Compile kernel once per worker before collecting timings (warmup).
	client.run(
		lambda: mandelbrot_chunk_early_exit(
			0, 8, 0, 8, 8, x_min, x_max, y_min, y_max, max_iter=10, use_early_exit=use_early_exit
		)
	)

	all_series = {}
	overall_best_time = (None, None, float("inf"))
	overall_best_lif = (None, None, float("inf"))

	try:
		for N in n_values:
			# Serial reference image for correctness checks.
			ref = mandelbrot_serial_blocks(
				N,
				x_min,
				x_max,
				y_min,
				y_max,
				max_iter=max_iter,
				n_chunks=serial_baseline_chunks,
				use_early_exit=use_early_exit,
			)
			# T1 baseline for LIF: serial execution with one chunk over the whole grid.
			t1_serial = _median_runtime(
				lambda: mandelbrot_serial_blocks(
					N,
					x_min,
					x_max,
					y_min,
					y_max,
					max_iter=max_iter,
					n_chunks=serial_baseline_chunks,
					use_early_exit=use_early_exit,
				),
				runs=n_runs,
			)

			n_chunks_values = [workers * mult for mult in chunk_multipliers]
			rows = []
			baseline_1x = None

			for n_chunks in n_chunks_values:
				def run_dask_once():
					# One Dask execution for this chunk count.
					return mandelbrot_dask(
						N,
						x_min,
						x_max,
						y_min,
						y_max,
						max_iter=max_iter,
						n_chunks=n_chunks,
						use_early_exit=use_early_exit,
					)

				# Compute once to validate numerical equivalence against serial reference.
				result = run_dask_once()
				if not np.array_equal(ref, result):
					raise AssertionError(
						f"Mismatch for N={N}, n_chunks={n_chunks}: Dask != serial reference"
					)

				# Median runtime (Tp) for this chunk setting.
				tp = _median_runtime(run_dask_once, runs=n_runs)

				# First iteration is the 1x baseline; capture it rather than measuring separately.
				if baseline_1x is None:
					baseline_1x = tp

				# Relative metrics against configured baselines.
				vs_1x = baseline_1x / tp
				speedup = t1_serial / tp
				lif = (workers * tp / t1_serial) - 1.0
				rows.append((n_chunks, tp, vs_1x, speedup, lif))

				if tp < overall_best_time[2]:
					overall_best_time = (N, n_chunks, tp)
				if lif < overall_best_lif[2]:
					overall_best_lif = (N, n_chunks, lif)

			all_series[N] = rows

			print(
				f"\nN={N} (serial T1, 1 chunk median over {n_runs} runs: {t1_serial:.4f} s; "
				f"16-worker 1x baseline: {baseline_1x:.4f} s; mode={mode})"
			)
			print("n chunks | time (s) | vs 1x | speedup | LIF")
			print("-" * 50)
			for n_chunks, tp, vs_1x, speedup, lif in rows:
				print(f"{n_chunks:8d} | {tp:8.4f} | {vs_1x:5.2f} | {speedup:7.2f} | {lif:7.3f}")

			best_time_row = min(rows, key=lambda r: r[1])
			best_lif_row = min(rows, key=lambda r: r[4])
			print(
				f"Optimal for N={N}: n_chunks={best_time_row[0]}, t_min={best_time_row[1]:.4f} s;\n"
				f"LIF_min={best_lif_row[4]:.3f} at n_chunks={best_lif_row[0]}"
			)

		fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
		for ax, N in zip(axes.flat, n_values):
			rows = all_series[N]
			xs = [r[0] for r in rows]
			wall_times = [r[1] for r in rows]
			speedups = [r[3] for r in rows]

			# Left axis: wall-clock time in seconds.
			line_time = ax.plot(xs, wall_times, marker="o", color="tab:blue", label="Wall time (s)")[0]

			# Right axis: dimensionless performance metrics.
			ax_r = ax.twinx()
			line_speedup = ax_r.plot(
				xs,
				speedups,
				marker="s",
				color="tab:green",
				label="Speedup (x, vs 16-worker 1x)",
			)[0]

			ax.set_xscale("log", base=2)
			ax.set_title(f"N={N}")
			ax.set_xlabel("n chunks")
			ax.set_ylabel("Wall time (s)", color="tab:blue")
			ax_r.set_ylabel("Speedup (x)", color="tab:green")
			ax.tick_params(axis="y", labelcolor="tab:blue")
			ax_r.tick_params(axis="y", labelcolor="tab:green")
			ax.grid(True, alpha=0.3)

			lines = [line_time, line_speedup]
			labels = [line.get_label() for line in lines]
			ax.legend(lines, labels, fontsize=8, loc="best")

		fig.suptitle(f"Dask Chunk Sweep ({mode}): Wall Time and Speedup", fontsize=12)
		plt.tight_layout()
		plt.savefig(f"dask chunk sweep_{mode}.png", dpi=160)

		print("\nOverall best time:")
		print(
			f"n_chunks optimal={overall_best_time[1]} (N={overall_best_time[0]}), "
			f"t_min={overall_best_time[2]:.4f} s"
		)
		print("Overall minimum LIF:")
		print(
			f"n_chunks optimal={overall_best_lif[1]} (N={overall_best_lif[0]}), "
			f"LIF_min={overall_best_lif[2]:.3f}"
		)
		print(f"Saved plot: dask chunk sweep_{mode}.png")
	finally:
		client.close()
		cluster.close()


def m2():
	print("\nRunning m2 without early exit...")
	_run_m2_case(use_early_exit=False)
	print("\nRunning m2 with early exit...")
	_run_m2_case(use_early_exit=True)

	"""
	Running m2 without early exit...
	
	N=1024 (serial T1, 1 chunk median over 5 runs: 0.0927 s; 16-worker 1x baseline: 0.0395 s; mode=without_early_exit)
	n chunks | time (s) | vs 1x | speedup | LIF
	--------------------------------------------------
		  16 |   0.0395 |  1.00 |    2.35 |   5.821
		  32 |   0.0470 |  0.84 |    1.97 |   7.116
		  64 |   0.0683 |  0.58 |    1.36 |  10.792
		 128 |   0.1063 |  0.37 |    0.87 |  17.341
		 256 |   0.1998 |  0.20 |    0.46 |  33.473
		 512 |   0.3791 |  0.10 |    0.24 |  64.429
	Optimal for N=1024: n_chunks=16, t_min=0.0395 s;
	LIF_min=5.821 at n_chunks=16
	
	N=2048 (serial T1, 1 chunk median over 5 runs: 0.6345 s; 16-worker 1x baseline: 0.7127 s; mode=without_early_exit)
	n chunks | time (s) | vs 1x | speedup | LIF
	--------------------------------------------------
		  16 |   0.7127 |  1.00 |    0.89 |  16.972
		  32 |   0.4427 |  1.61 |    1.43 |  10.163
		  64 |   0.0993 |  7.18 |    6.39 |   1.504
		 128 |   0.1369 |  5.20 |    4.63 |   2.453
		 256 |   0.2215 |  3.22 |    2.86 |   4.585
		 512 |   0.4323 |  1.65 |    1.47 |   9.901
	Optimal for N=2048: n_chunks=64, t_min=0.0993 s;
	LIF_min=1.504 at n_chunks=64
	
	N=4096 (serial T1, 1 chunk median over 5 runs: 2.6747 s; 16-worker 1x baseline: 3.0432 s; mode=without_early_exit)
	n chunks | time (s) | vs 1x | speedup | LIF
	--------------------------------------------------
		  16 |   3.0432 |  1.00 |    0.88 |  17.205
		  32 |   2.9534 |  1.03 |    0.91 |  16.667
		  64 |   2.8886 |  1.05 |    0.93 |  16.280
		 128 |   2.0054 |  1.52 |    1.33 |  10.996
		 256 |   0.4186 |  7.27 |    6.39 |   1.504
		 512 |   0.6504 |  4.68 |    4.11 |   2.891
	Optimal for N=4096: n_chunks=256, t_min=0.4186 s;
	LIF_min=1.504 at n_chunks=256
	
	N=8192 (serial T1, 1 chunk median over 5 runs: 10.3945 s; 16-worker 1x baseline: 11.9172 s; mode=without_early_exit)
	n chunks | time (s) | vs 1x | speedup | LIF
	--------------------------------------------------
		  16 |  11.9172 |  1.00 |    0.87 |  17.344
		  32 |  11.6794 |  1.02 |    0.89 |  16.978
		  64 |  11.5140 |  1.04 |    0.90 |  16.723
		 128 |  11.6544 |  1.02 |    0.89 |  16.939
		 256 |  11.5162 |  1.03 |    0.90 |  16.727
		 512 |   7.1284 |  1.67 |    1.46 |   9.973
	Optimal for N=8192: n_chunks=512, t_min=7.1284 s;
	LIF_min=9.973 at n_chunks=512
	
	Overall best time:
	n_chunks optimal=16 (N=1024), t_min=0.0395 s
	Overall minimum LIF:
	n_chunks optimal=64 (N=2048), LIF_min=1.504
	Saved plot: dask chunk sweep_without_early_exit.png
	
	Running m2 with early exit...
	
	N=1024 (serial T1, 1 chunk median over 5 runs: 0.0933 s; 16-worker 1x baseline: 0.0417 s; mode=with_early_exit)
	n chunks | time (s) | vs 1x | speedup | LIF
	--------------------------------------------------
		  16 |   0.0417 |  1.00 |    2.24 |   6.142
		  32 |   0.0466 |  0.89 |    2.00 |   6.995
		  64 |   0.0694 |  0.60 |    1.34 |  10.906
		 128 |   0.1115 |  0.37 |    0.84 |  18.110
		 256 |   0.2151 |  0.19 |    0.43 |  35.887
		 512 |   0.3972 |  0.10 |    0.23 |  67.104
	Optimal for N=1024: n_chunks=16, t_min=0.0417 s;
	LIF_min=6.142 at n_chunks=16
	
	N=2048 (serial T1, 1 chunk median over 5 runs: 0.5878 s; 16-worker 1x baseline: 0.5332 s; mode=with_early_exit)
	n chunks | time (s) | vs 1x | speedup | LIF
	--------------------------------------------------
		  16 |   0.5332 |  1.00 |    1.10 |  13.515
		  32 |   0.2181 |  2.45 |    2.70 |   4.936
		  64 |   0.0940 |  5.67 |    6.25 |   1.559
		 128 |   0.1310 |  4.07 |    4.49 |   2.565
		 256 |   0.2347 |  2.27 |    2.50 |   5.390
		 512 |   0.4248 |  1.26 |    1.38 |  10.563
	Optimal for N=2048: n_chunks=64, t_min=0.0940 s;
	LIF_min=1.559 at n_chunks=64
	
	N=4096 (serial T1, 1 chunk median over 5 runs: 2.5561 s; 16-worker 1x baseline: 2.2384 s; mode=with_early_exit)
	n chunks | time (s) | vs 1x | speedup | LIF
	--------------------------------------------------
		  16 |   2.2384 |  1.00 |    1.14 |  13.012
		  32 |   1.7635 |  1.27 |    1.45 |  10.039
		  64 |   1.3425 |  1.67 |    1.90 |   7.404
		 128 |   0.3491 |  6.41 |    7.32 |   1.185
		 256 |   0.3326 |  6.73 |    7.69 |   1.082
		 512 |   0.4930 |  4.54 |    5.18 |   2.086
	Optimal for N=4096: n_chunks=256, t_min=0.3326 s;
	LIF_min=1.082 at n_chunks=256
	
	N=8192 (serial T1, 1 chunk median over 5 runs: 10.4563 s; 16-worker 1x baseline: 8.9900 s; mode=with_early_exit)
	n chunks | time (s) | vs 1x | speedup | LIF
	--------------------------------------------------
		  16 |   8.9900 |  1.00 |    1.16 |  12.756
		  32 |   7.2128 |  1.25 |    1.45 |  10.037
		  64 |   5.7618 |  1.56 |    1.81 |   7.817
		 128 |   5.5046 |  1.63 |    1.90 |   7.423
		 256 |   4.4705 |  2.01 |    2.34 |   5.841
		 512 |   1.1052 |  8.13 |    9.46 |   0.691
	Optimal for N=8192: n_chunks=512, t_min=1.1052 s;
	LIF_min=0.691 at n_chunks=512
	
	Overall best time:
	n_chunks optimal=16 (N=1024), t_min=0.0417 s
	Overall minimum LIF:
	n_chunks optimal=512 (N=8192), LIF_min=0.691
	Saved plot: dask chunk sweep_with_early_exit.png
	"""


if __name__ == "__main__":
	# m1()
	m2()
