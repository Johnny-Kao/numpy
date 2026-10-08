"""Focused correctness probe for NumPy searchsorted research.

Runs against the locally installed *built* NumPy; never downloads wheels as substitutes.
"""
import numpy as np

print("numpy", np.__version__, np.__file__)
rng = np.random.default_rng(32895)
cases = 0
for dtype in (np.int32, np.int64, np.float32, np.float64):
    for n in (0, 1, 2, 3, 7, 16, 31, 32, 127, 1024):
        for q in (0, 1, 2, 8, 32, 257):
            a = np.sort(rng.integers(-12, 13, size=n).astype(dtype))
            if np.issubdtype(dtype, np.floating):
                a = np.sort(np.r_[a, [-np.inf, np.inf, np.nan]].astype(dtype))
                queries = np.r_[rng.integers(-14, 15, size=q), [-np.inf, -0.0, 0.0, np.inf, np.nan]].astype(dtype)
            else:
                queries = rng.integers(-14, 15, size=q).astype(dtype)
            for side in ("left", "right"):
                reference = np.array([np.searchsorted(a, value, side=side) for value in queries], dtype=np.intp)
                actual = np.searchsorted(a, queries, side=side)
                np.testing.assert_array_equal(actual, reference)
                cases += 1
                if len(a) > 1:
                    astrided = np.repeat(a, 2)[::2]
                    np.testing.assert_array_equal(np.searchsorted(astrided, queries, side=side), reference)
                    cases += 1
print(f"Correctness PASS: {cases} checks")
