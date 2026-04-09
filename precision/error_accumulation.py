import numpy as np

n_values = [10, 100, 1_000, 10_000, 100_000]

if __name__ == "__main__":
    for dtype in [np.float32, np.float64]:
        print(f"\n{dtype.__name__}:")
        for n in n_values:
            total = dtype(0.0)
            for _ in range(n):
                total += dtype(0.1)
            expected = n * 0.1
            rel_error = abs(float(total) - expected) / expected
            print(f" n={n:>7d}: result={float(total):.10f} rel_error={rel_error:.2e}")

    """
    float32:
     n=     10: result=1.0000001192 rel_error=1.19e-07
     n=    100: result=10.0000019073 rel_error=1.91e-07
     n=   1000: result=99.9990463257 rel_error=9.54e-06
     n=  10000: result=999.9028930664 rel_error=9.71e-05
     n= 100000: result=9998.5566406250 rel_error=1.44e-04
    
    float64:
     n=     10: result=1.0000000000 rel_error=1.11e-16
     n=    100: result=10.0000000000 rel_error=1.95e-15
     n=   1000: result=100.0000000000 rel_error=1.41e-14
     n=  10000: result=1000.0000000002 rel_error=1.59e-13
     n= 100000: result=10000.0000000188 rel_error=1.88e-12
    """
