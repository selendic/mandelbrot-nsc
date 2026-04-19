import cProfile
import pstats


def main():
    """Run cProfile on naive and NumPy Mandelbrot implementations and print top stats."""
    cProfile.run("""mandelbrot_time_test(
        func_gen=mandelbrot_naive.generate_complex_grid,
        func_calc=mandelbrot_naive.compute_mandelbrot,
        start_size_log_2=2,
        top_size_log_2=2,
        n_runs_per_size=1,
        show_plots=False
    )""", "naive_profile.prof")
    cProfile.run("""mandelbrot_time_test(
        func_gen=mandelbrot_numpy.generate_complex_grid,
        func_calc=mandelbrot_numpy.compute_mandelbrot,
        start_size_log_2=2,
        top_size_log_2=2,
        n_runs_per_size=1,
        show_plots=False
    )""", "numpy_profile.prof")

    for profile_name in ["naive_profile.prof", "numpy_profile.prof"]:
        print(f"\nProfile for {profile_name}:\n")
        stats = pstats.Stats(profile_name)
        stats.sort_stats(pstats.SortKey.CUMULATIVE)
        stats.print_stats(10)


if __name__ == "__main__":
    main()
