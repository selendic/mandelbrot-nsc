import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.strategies import complex_numbers

from cpu_parallelization import mandelbrot_numba_parallel
from naive import mandelbrot_naive
from numba_jit import mandelbrot_numba_jit
from numpy_simd import mandelbrot_numpy

KNOWN_CASES = [
    (0 + 0j, 100, [100]),  # origin: never escapes
    (5.0 + 0j, 100, [0, 1]),  # far outside, escapes on iteration 0/1
    (-2.5 + 0j, 100, [0, 1]),  # left tip of set
]
IMPLEMENTATIONS_POINT = [
    mandelbrot_naive.mandelbrot_point,
    mandelbrot_numba_jit.mandelbrot_point_numba,
    mandelbrot_numba_parallel.mandelbrot_point
]
IMPLEMENTATIONS_FULL = [
    mandelbrot_naive.compute_mandelbrot,
    mandelbrot_numpy.compute_mandelbrot,
    mandelbrot_numba_jit.mandelbrot_naive_full_numba,
]


@pytest.mark.parametrize("impl", IMPLEMENTATIONS_POINT)
@pytest.mark.parametrize("c, max_iter, expected", KNOWN_CASES)
def test_pixel_all(impl, c, max_iter, expected):
    assert impl(c, max_iter=max_iter) in expected


@pytest.mark.parametrize("impl", IMPLEMENTATIONS_FULL)
def test_full_all(impl):
    grid = mandelbrot_numba_jit.generate_complex_grid(256)
    expected = mandelbrot_naive.compute_mandelbrot(
        grid, max_iter=100
    )
    result = impl(grid, max_iter=100)

    np.testing.assert_array_equal(result, expected)


# Draw random points with |c| <= 3 (covers both inside and outside the set)
@given(complex_numbers(max_magnitude=3.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=200)
@pytest.mark.parametrize("impl", IMPLEMENTATIONS_POINT)
def test_result_in_range(impl, c):
    assert 0 <= impl(c, max_iter=100) <= 100


# Draw random points far outside the set (|c| between 3 and 10)
@given(complex_numbers(min_magnitude=3.0, max_magnitude=10.0, allow_nan=False, allow_infinity=False))
@pytest.mark.parametrize("impl", IMPLEMENTATIONS_POINT)
def test_outside_set_escapes(impl, c):
    assert impl(c, max_iter=100) < 100

# ============================================= test session starts ==============================================
# platform linux -- Python 3.11.14, pytest-9.0.2, pluggy-1.6.0 -- /home/peppermint/miniforge3/envs/nsc-2026/bin/python3.11
# cachedir: .pytest_cache
# hypothesis profile 'default'
# rootdir: /home/peppermint/Aalborg/CE8/numerical_scientific_computing/mandelbrot-nsc
# plugins: anyio-4.13.0, cov-7.1.0, hypothesis-6.152.1
# collected 18 items
#
# tests/test_mandelbrot.py::test_pixel_all[0j-100-expected0-mandelbrot_point0] PASSED                      [  5%]
# tests/test_mandelbrot.py::test_pixel_all[0j-100-expected0-mandelbrot_point_numba] PASSED                 [ 11%]
# tests/test_mandelbrot.py::test_pixel_all[0j-100-expected0-mandelbrot_point1] PASSED                      [ 16%]
# tests/test_mandelbrot.py::test_pixel_all[(5+0j)-100-expected1-mandelbrot_point0] PASSED                  [ 22%]
# tests/test_mandelbrot.py::test_pixel_all[(5+0j)-100-expected1-mandelbrot_point_numba] PASSED             [ 27%]
# tests/test_mandelbrot.py::test_pixel_all[(5+0j)-100-expected1-mandelbrot_point1] PASSED                  [ 33%]
# tests/test_mandelbrot.py::test_pixel_all[(-2.5+0j)-100-expected2-mandelbrot_point0] PASSED               [ 38%]
# tests/test_mandelbrot.py::test_pixel_all[(-2.5+0j)-100-expected2-mandelbrot_point_numba] PASSED          [ 44%]
# tests/test_mandelbrot.py::test_pixel_all[(-2.5+0j)-100-expected2-mandelbrot_point1] PASSED               [ 50%]
# tests/test_mandelbrot.py::test_full_all[compute_mandelbrot0] PASSED                                      [ 55%]
# tests/test_mandelbrot.py::test_full_all[compute_mandelbrot1] PASSED                                      [ 61%]
# tests/test_mandelbrot.py::test_full_all[mandelbrot_naive_full_numba] PASSED                              [ 66%]
# tests/test_mandelbrot.py::test_result_in_range[mandelbrot_point0] PASSED                                 [ 72%]
# tests/test_mandelbrot.py::test_result_in_range[mandelbrot_point_numba] PASSED                            [ 77%]
# tests/test_mandelbrot.py::test_result_in_range[mandelbrot_point1] PASSED                                 [ 83%]
# tests/test_mandelbrot.py::test_outside_set_escapes[mandelbrot_point0] PASSED                             [ 88%]
# tests/test_mandelbrot.py::test_outside_set_escapes[mandelbrot_point_numba] PASSED                        [ 94%]
# tests/test_mandelbrot.py::test_outside_set_escapes[mandelbrot_point1] PASSED                             [100%]
#
# ================================================ tests coverage ================================================
# _______________________________ coverage: platform linux, python 3.11.14-final-0 _______________________________
#
# Name                                               Stmts   Miss  Cover
# ----------------------------------------------------------------------
# cpu_parallelization/__init__.py                        0      0   100%
# cpu_parallelization/mandelbrot_numba_parallel.py     281    260     7%
# cpu_parallelization/map_filter_reduce.py              28     28     0%
# cpu_parallelization/monte_carlo_pi_parallel.py       163    163     0%
# cpu_parallelization/monte_carlo_pi_serial.py          26     26     0%
# dasked/__init__.py                                     0      0   100%
# dasked/mandelbrot.py                                 205    205     0%
# dasked/monte_carlo_pi.py                              78     78     0%
# naive/__init__.py                                      0      0   100%
# naive/mandelbrot_naive.py                             41     14    66%
# numba_jit/__init__.py                                  0      0   100%
# numba_jit/m1_cprofile_naive_and_numpy.py              12     12     0%
# numba_jit/m2_line_level_profiling.py                   4      4     0%
# numba_jit/m4_32bit_vs_64_bit.py                       23     23     0%
# numba_jit/m5.py                                       17     17     0%
# numba_jit/mandelbrot_numba_jit.py                    130    105    19%
# numpy_simd/__init__.py                                 0      0   100%
# numpy_simd/mandelbrot_numpy.py                        48     25    48%
# numpy_simd/milestone_3.py                             25     25     0%
# ----------------------------------------------------------------------
# TOTAL                                               1081    985     9%
# ============================================== 18 passed in 3.65s ==============================================
