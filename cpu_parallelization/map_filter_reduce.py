import random, time
from functools import reduce
from multiprocessing import Pool

N = 1_000_000


def subtract_seven(x):
    return x - 7


def map_filter_reduce_pipeline_serial(data):
    t0 = time.perf_counter()
    result_ser = reduce(lambda a, b: a + b,
                        filter(lambda x: x % 2 == 1,
                               map(subtract_seven, data)
                               )
                        )
    t_serial = time.perf_counter() - t0
    return result_ser, t_serial


def map_filter_reduce_pipeline_parallel(data):
    with Pool() as pool:
        t0 = time.perf_counter()
        mapped = pool.map(subtract_seven, data)
        result_par = reduce(lambda a, b: a + b,
                        filter(lambda x: x % 2 == 1, mapped))
        t_parallel = time.perf_counter() - t0
    return result_par, t_parallel


def main():
    data = [random.randint(10, 100) for _ in range(N)]
    result_ser, t_serial = map_filter_reduce_pipeline_serial(data)
    result_par, t_parallel = map_filter_reduce_pipeline_parallel(data)
    print(f"Serial:   {t_serial:.4f}s  (result={result_ser})")
    print(f"Parallel: {t_parallel:.4f}s  (result={result_par})")
    print(f"Speedup:  {t_serial / t_parallel:.2f}x")


if __name__ == "__main__":
    main()

    """
    Serial:   0.0612s  (result=24279241)
    Parallel: 0.0691s  (result=24279241)
    Speedup:  0.89x
    """
