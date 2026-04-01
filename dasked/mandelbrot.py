from dask import delayed
from dask.distributed import Client, LocalCluster
import dask, numpy as np, time, statistics
import math
import matplotlib.pyplot as plt
from numba import njit

REMOTE_CLIENT_STR = "tcp://10.92.1.177:8786"
NUM_RUNS = 5


@njit(cache=True)
def mandelbrot_chunk_early_exit(row_start: int, row_end: int,
								col_start: int, col_end: int, N: int,
								x_min: float, x_max: float, y_min: float, y_max: float,
								threshold=2.0, max_iter=100, use_early_exit=True):
	"""
	Compute Mandelbrot iteration counts for one tile of the full grid.

	Parameters
	----------
	row_start : int
		Starting row index (inclusive) in the full image.
	row_end : int
		Ending row index (exclusive) in the full image.
	col_start : int
		Starting column index (inclusive) in the full image.
	col_end : int
		Ending column index (exclusive) in the full image.
	N : int
		Full image resolution (N x N), used to map indices to complex-plane coordinates.
	x_min : float
		Minimum x-coordinate (real axis) of the sampled domain.
	x_max : float
		Maximum x-coordinate (real axis) of the sampled domain.
	y_min : float
		Minimum y-coordinate (imaginary axis) of the sampled domain.
	y_max : float
		Maximum y-coordinate (imaginary axis) of the sampled domain.
	threshold : float
		Escape radius threshold (default is 2.0).
	max_iter : int
		Maximum number of iterations per point (default is 100).
	use_early_exit : bool
		If True, stop iterating once all points in the tile have escaped.

	Returns
	-------
	np.ndarray
		A ``(row_end - row_start, col_end - col_start)`` int32 array of iteration counts.
	"""
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
	"""
	Compute the Mandelbrot image using Dask-delayed square-tile tasks.

	Parameters
	----------
	N : int
		Square image resolution (N x N).
	x_min : float
		Minimum x-coordinate (real axis) of the sampled domain.
	x_max : float
		Maximum x-coordinate (real axis) of the sampled domain.
	y_min : float
		Minimum y-coordinate (imaginary axis) of the sampled domain.
	y_max : float
		Maximum y-coordinate (imaginary axis) of the sampled domain.
	max_iter : int
		Maximum number of iterations per point (default is 100).
	n_chunks : int
		Target number of tiles. Tiles are arranged as a square grid using
		``isqrt(n_chunks)`` chunks per axis.
	use_early_exit : bool
		If True, tile kernels stop when all points in a tile have escaped.

	Returns
	-------
	np.ndarray
		A ``(N, N)`` int32 array containing Mandelbrot iteration counts.
	"""
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
	"""
	Compute a serial Mandelbrot baseline using the same square-tile layout as Dask.

	Parameters
	----------
	N : int
		Square image resolution (N x N).
	x_min : float
		Minimum x-coordinate (real axis) of the sampled domain.
	x_max : float
		Maximum x-coordinate (real axis) of the sampled domain.
	y_min : float
		Minimum y-coordinate (imaginary axis) of the sampled domain.
	y_max : float
		Maximum y-coordinate (imaginary axis) of the sampled domain.
	max_iter : int
		Maximum number of iterations per point (default is 100).
	n_chunks : int
		Target number of tiles used to derive the square tiling layout.
	use_early_exit : bool
		If True, tile kernels stop when all points in a tile have escaped.

	Returns
	-------
	np.ndarray
		A ``(N, N)`` int32 array containing Mandelbrot iteration counts.
	"""
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


def m1(N: int = 1024, n_chunks: int = None):
	"""
	Run a correctness check and timing benchmark for the Dask Mandelbrot path.

	Parameters
	----------
	N : int
		Square image resolution (N x N) used for verification and timing.
	n_chunks : int or None
		Number of chunks to use for Dask execution. If None, defaults to the
		number of connected cluster workers.

	Returns
	-------
	None
		Prints correctness and timing information to stdout.
	"""
	max_iter = 100
	X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25
	# cluster = LocalCluster(n_workers=8, threads_per_worker=1)
	# client = Client(cluster) ->
	client = Client(REMOTE_CLIENT_STR)
	print(client)
	workers = len(client.scheduler_info(-1)["workers"])
	if workers == 0:
		raise RuntimeError("Remote Dask cluster has no connected workers.")
	if n_chunks is None:
		n_chunks = workers
	# warm up all workers
	client.run(lambda: mandelbrot_chunk_early_exit(0, 8, 0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter=10))

	# Reference result from one full-grid kernel call.
	ref = mandelbrot_chunk_early_exit(0, N, 0, N, N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter=max_iter)
	# Dask-computed result to compare against the reference.
	result = mandelbrot_dask(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter, n_chunks=n_chunks)
	if not np.array_equal(ref, result):
		raise AssertionError("Dask result does not match serial reference output.")

	times = []
	for _ in range(NUM_RUNS):
		# Measure one full Dask run.
		t0 = time.perf_counter()
		result = mandelbrot_dask(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter, n_chunks=n_chunks)
		times.append(time.perf_counter() - t0)
	print("Verification passed: dask output matches serial output.")
	print(f"Dask local (N={N}, {NUM_RUNS} runs, n_chunks={n_chunks}): median {statistics.median(times):.3f} s")
	client.close()
	# cluster.close()

	"""
	Verification passed: dask output matches serial output.
	Dask local (N=1024, 3 runs, n_chunks=32): median 0.046 s
	"""


def _median_runtime(func, runs=3):
	"""
	Measure and return the median runtime of a callable.

	Parameters
	----------
	func : callable
		Zero-argument function to execute and time.
	runs : int
		Number of executions used to compute the median runtime.

	Returns
	-------
	float
		Median elapsed time in seconds across ``runs`` executions.
	"""
	times = []
	for _ in range(runs):
		# Time a single call and append elapsed seconds.
		t0 = time.perf_counter()
		func()
		times.append(time.perf_counter() - t0)
	return statistics.median(times)


def _run_m2_case(use_early_exit):
	"""
	Run one milestone-2 chunk-sweep experiment for a selected early-exit mode.

	Parameters
	----------
	use_early_exit : bool
		Whether tile kernels should stop early when all points in a tile escape.

	Returns
	-------
	None
		Prints benchmark tables and saves a chunk-sweep plot image.
	"""
	n_values = [1024, 2048, 4096, 8192, 16384]
	chunk_multipliers = [1, 2, 4, 8, 16, 32, 64, 128]
	max_iter = 100
	x_min, x_max, y_min, y_max = -2.5, 1.0, -1.25, 1.25
	serial_baseline_chunks = 1
	mode = "with_early_exit" if use_early_exit else "no_early_exit"

	# cluster = LocalCluster(n_workers=16, threads_per_worker=1)
	# client = Client(cluster)
	client = Client(REMOTE_CLIENT_STR)
	print(client)
	workers = len(client.scheduler_info(-1)["workers"])
	if workers == 0:
		raise RuntimeError("Remote Dask cluster has no connected workers.")

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
				runs=NUM_RUNS,
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
				tp = _median_runtime(run_dask_once, runs=NUM_RUNS)

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
				f"\nN={N} (serial T1, 1 chunk median over {NUM_RUNS} runs: {t1_serial:.4f} s; "
				f"{workers}-worker 1x baseline: {baseline_1x:.4f} s; mode={mode})"
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
		for ax, N in zip(axes.flat, n_values[1:]):  # on Strato it is more interesting to look at 16k than 1k
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
				label=f"Speedup (x, vs {workers}-worker 1x)",
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
		plt.savefig(f"dask_chunk_sweep_strato_{mode}.png", dpi=160)

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
		print(f"Saved plot: dask_chunk_sweep_strato_{mode}.png")
	finally:
		client.close()
		# cluster.close()


def m2():
	"""
	Entry point for the milestone-2 Dask chunk-sweep experiment.

	Returns
	-------
	None
		Runs the configured experiment mode and prints benchmark results.
	"""
	# print("\nRunning m2 without early exit...")
	# _run_m2_case(use_early_exit=False)
	print("\nRunning m2 with early exit...")
	_run_m2_case(use_early_exit=True)

	"""
	Local:
	
	Running m2 no early exit...
	
	N=1024 (serial T1, 1 chunk median over 5 runs: 0.0927 s; 16-worker 1x baseline: 0.0395 s; mode=no_early_exit)
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
	
	N=2048 (serial T1, 1 chunk median over 5 runs: 0.6345 s; 16-worker 1x baseline: 0.7127 s; mode=no_early_exit)
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
	
	N=4096 (serial T1, 1 chunk median over 5 runs: 2.6747 s; 16-worker 1x baseline: 3.0432 s; mode=no_early_exit)
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
	
	N=8192 (serial T1, 1 chunk median over 5 runs: 10.3945 s; 16-worker 1x baseline: 11.9172 s; mode=no_early_exit)
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
	Saved plot: dask_chunk_sweep_local_no_early_exit.png
	
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
	Saved plot: dask_chunk_sweep_local_with_early_exit.png
	"""
	###################################################################################################################


def plot_worker_scaling():
	"""
	Plot recorded worker-scaling timing results for a fixed benchmark setup.

	Returns
	-------
	None
		Saves the figure as ``worker_scaling_N=4096.png``.
	"""
	data = {
		4:  [0.700, 0.718, 0.732, 0.703],
		8:  [0.621, 0.592, 0.626, 0.650],
		12: [0.515, 0.538, 0.529, 0.527],
		16: [0.484, 0.499, 0.490, 0.493],
	}

	workers = []
	times = []
	avgs = []

	for w, runs in data.items():
		workers.extend([w] * len(runs))
		times.extend(runs)
		avgs.append((w, np.mean(runs)))

	avg_x, avg_y = zip(*avgs)

	fig, ax = plt.subplots(figsize=(8, 5))

	ax.scatter(workers, times, color="steelblue", alpha=0.6, zorder=2, label="Individual medians")
	ax.plot(avg_x, avg_y, color="tomato", marker="D", linewidth=2, markersize=7, zorder=3, label="Median avg")

	ax.set_xlabel("Worker count")
	ax.set_ylabel("Time (s)")
	ax.set_title("Dask Worker Scaling (N=4096, chunks=128)")
	ax.set_xticks(list(data.keys()))
	ax.legend()
	ax.grid(True, linestyle="--", alpha=0.4)

	plt.tight_layout()
	plt.savefig("worker_scaling_N=4096.png", dpi=300)


if __name__ == "__main__":
	# m1(N=4096, n_chunks=128)
	# m2()
	plot_worker_scaling()

	#######################
	"""
	ssh -i ~/.ssh/id_ed25519 ubuntu@10.92.1.104/245/121/42
	/opt/miniconda3/condabin/conda init
	( conda activate nsc-2026 )
	dask worker 10.92.1.177:8786 --nworkers -1 --nthreads 1
	"""
	#######################
	"""
	Strato:
	
	Running m2 with early exit...
	<Client: 'tcp://10.92.1.177:8786' processes=16 threads=16, memory=46.72 GiB>
	
	N=1024 (serial T1, 1 chunk median over 5 runs: 0.1417 s; 16-worker 1x baseline: 0.0813 s; mode=with_early_exit)
	n chunks | time (s) | vs 1x | speedup | LIF
	--------------------------------------------------
		  16 |   0.0813 |  1.00 |    1.74 |   8.179
		  32 |   0.1019 |  0.80 |    1.39 |  10.505
		  64 |   0.1647 |  0.49 |    0.86 |  17.601
		 128 |   0.2737 |  0.30 |    0.52 |  29.900
		 256 |   0.5553 |  0.15 |    0.26 |  61.697
		 512 |   1.0045 |  0.08 |    0.14 | 112.419
		1024 |   2.1876 |  0.04 |    0.06 | 246.012
		2048 |   4.6042 |  0.02 |    0.03 | 518.875
	Optimal for N=1024: n_chunks=16, t_min=0.0813 s;
	LIF_min=8.179 at n_chunks=16
	
	N=2048 (serial T1, 1 chunk median over 5 runs: 0.6446 s; 16-worker 1x baseline: 0.2597 s; mode=with_early_exit)
	n chunks | time (s) | vs 1x | speedup | LIF
	--------------------------------------------------
		  16 |   0.2597 |  1.00 |    2.48 |   5.445
		  32 |   0.1961 |  1.32 |    3.29 |   3.866
		  64 |   0.2162 |  1.20 |    2.98 |   4.366
		 128 |   0.3211 |  0.81 |    2.01 |   6.970
		 256 |   0.5861 |  0.44 |    1.10 |  13.547
		 512 |   1.1190 |  0.23 |    0.58 |  26.773
		1024 |   2.2592 |  0.11 |    0.29 |  55.073
		2048 |   4.6779 |  0.06 |    0.14 | 115.106
	Optimal for N=2048: n_chunks=32, t_min=0.1961 s;
	LIF_min=3.866 at n_chunks=32
	
	N=4096 (serial T1, 1 chunk median over 5 runs: 2.3864 s; 16-worker 1x baseline: 0.6297 s; mode=with_early_exit)
	n chunks | time (s) | vs 1x | speedup | LIF
	--------------------------------------------------
		  16 |   0.6297 |  1.00 |    3.79 |   3.222
		  32 |   0.5837 |  1.08 |    4.09 |   2.913
		  64 |   0.7005 |  0.90 |    3.41 |   3.697
		 128 |   0.4801 |  1.31 |    4.97 |   2.219
		 256 |   0.7407 |  0.85 |    3.22 |   3.966
		 512 |   1.2157 |  0.52 |    1.96 |   7.151
		1024 |   2.4399 |  0.26 |    0.98 |  15.359
		2048 |   4.8638 |  0.13 |    0.49 |  31.611
	Optimal for N=4096: n_chunks=128, t_min=0.4801 s;
	LIF_min=2.219 at n_chunks=128
	
	N=8192 (serial T1, 1 chunk median over 5 runs: 8.8606 s; 16-worker 1x baseline: 2.8380 s; mode=with_early_exit)
	n chunks | time (s) | vs 1x | speedup | LIF
	--------------------------------------------------
		  16 |   2.8380 |  1.00 |    3.12 |   4.125
		  32 |   2.1289 |  1.33 |    4.16 |   2.844
		  64 |   1.9981 |  1.42 |    4.43 |   2.608
		 128 |   1.6048 |  1.77 |    5.52 |   1.898
		 256 |   1.5801 |  1.80 |    5.61 |   1.853
		 512 |   1.7530 |  1.62 |    5.05 |   2.165
		1024 |   2.8546 |  0.99 |    3.10 |   4.155
		2048 |   5.4319 |  0.52 |    1.63 |   8.809
	Optimal for N=8192: n_chunks=256, t_min=1.5801 s;
	LIF_min=1.853 at n_chunks=256
	
	N=16384 (serial T1, 1 chunk median over 5 runs: 32.6217 s; 16-worker 1x baseline: 10.5030 s; mode=with_early_exit)
	n chunks | time (s) | vs 1x | speedup | LIF
	--------------------------------------------------
		  16 |  10.5030 |  1.00 |    3.11 |   4.151
		  32 |   9.9972 |  1.05 |    3.26 |   3.903
		  64 |   9.0471 |  1.16 |    3.61 |   3.437
		 128 |   6.2439 |  1.68 |    5.22 |   2.062
		 256 |   5.9519 |  1.76 |    5.48 |   1.919
		 512 |   5.7058 |  1.84 |    5.72 |   1.799
		1024 |   5.7816 |  1.82 |    5.64 |   1.836
		2048 |   7.8862 |  1.33 |    4.14 |   2.868
	Optimal for N=16384: n_chunks=512, t_min=5.7058 s;
	LIF_min=1.799 at n_chunks=512
	
	Overall best time:
	n_chunks optimal=16 (N=1024), t_min=0.0813 s
	Overall minimum LIF:
	n_chunks optimal=512 (N=16384), LIF_min=1.799
	Saved plot: dask_chunk_sweep_strato_with_early_exit.png
	"""

	"""
	Worker scaling for best chunk size at given resolution (N=4096: chunks=128;2^3x16) 
	______________________________________________________________
	1 instance (4 workers):
	<Client: 'tcp://10.92.1.177:8786' processes=4 threads=4, memory=11.68 GiB>
	Verification passed: dask output matches serial output.
	- Dask local (N=4096, 5 runs, n_chunks=128): median 0.700 s
	- Dask local (N=4096, 5 runs, n_chunks=128): median 0.718 s
	- Dask local (N=4096, 5 runs, n_chunks=128): median 0.732 s
	- Dask local (N=4096, 5 runs, n_chunks=128): median 0.703 s

	2 instances (8 workers):
	<Client: 'tcp://10.92.1.177:8786' processes=8 threads=8, memory=23.36 GiB>
	Verification passed: dask output matches serial output.
	- Dask local (N=4096, 5 runs, n_chunks=128): median 0.621 s
	- Dask local (N=4096, 5 runs, n_chunks=128): median 0.592 s
	- Dask local (N=4096, 5 runs, n_chunks=128): median 0.626 s
	- Dask local (N=4096, 5 runs, n_chunks=128): median 0.650 s

	3 instances (12 workers):
	<Client: 'tcp://10.92.1.177:8786' processes=12 threads=12, memory=35.04 GiB>
	Verification passed: dask output matches serial output.
	- Dask local (N=4096, 5 runs, n_chunks=128): median 0.515 s
	- Dask local (N=4096, 5 runs, n_chunks=128): median 0.538 s
	- Dask local (N=4096, 5 runs, n_chunks=128): median 0.529 s
	- Dask local (N=4096, 5 runs, n_chunks=128): median 0.527 s

	4 instances (16 workers):
	<Client: 'tcp://10.92.1.177:8786' processes=16 threads=16, memory=46.72 GiB>
	Verification passed: dask output matches serial output.
	- Dask local (N=4096, 5 runs, n_chunks=128): median 0.484 s
	- Dask local (N=4096, 5 runs, n_chunks=128): median 0.499 s
	- Dask local (N=4096, 5 runs, n_chunks=128): median 0.490 s
	- Dask local (N=4096, 5 runs, n_chunks=128): median 0.493 s
	"""
