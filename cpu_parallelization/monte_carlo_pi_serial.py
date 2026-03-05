import math, random, time, statistics

NUM_SAMPLES = 10_000_000
NUM_RUNS = 10


def estimate_pi_serial(num_samples):
    inside_circle = 0
    for _ in range(num_samples):
        x, y = random.random(), random.random()
        if x * x + y * y <= 1:
            inside_circle += 1
    return 4 * inside_circle / num_samples


if __name__ == "__main__":
    pi_estimates = []
    times = []
    for _ in range(NUM_RUNS):
        t = time.perf_counter()
        pi_estimates.append(estimate_pi_serial(NUM_SAMPLES))
        times.append(time.perf_counter() - t)
    t_serial_med = statistics.median(times)
    pi_estimate_mean = statistics.mean(pi_estimates)
    pi_estimate_std = statistics.stdev(pi_estimates)
    print(f"Num runs:         {NUM_RUNS}")
    print(f"Num samples:      {NUM_SAMPLES}.")
    print(f"PI estimate:      {pi_estimate_mean:.6f}±{pi_estimate_std:.6f}")
    print(f"Error (for mean): {abs(pi_estimate_mean - math.pi):.6f}")
    print(f"Serial time:      {t_serial_med:.3f}s")

    """
    Num runs:         10
    Num samples:      10000000
    PI estimate:      3.141579±0.000538
    Error (for mean): 0.000014
    Serial time:      0.748s
    """
