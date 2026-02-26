from util import mandelbrot_time_test
from naive.mandelbrot_naive import generate_complex_grid, compute_mandelbrot

def main():

    mandelbrot_time_test(
        func_gen=generate_complex_grid,
        func_calc=compute_mandelbrot,
        start_size_log_2=2,
        top_size_log_2=2,
        n_runs_per_size=1,
        show_plots=False
    )
