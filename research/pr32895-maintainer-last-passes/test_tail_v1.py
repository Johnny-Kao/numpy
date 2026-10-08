"""Correctness smoke suite for the research-only last-two-passes experiment."""
import numpy as np

rng = np.random.default_rng(32895)
tested = 0
for dtype in (np.int32, np.int64, np.float32, np.float64):
    for n in (0, 1, 2, 3, 7, 16, 31, 256, 4096):
        source = np.sort(rng.integers(-20, 20, size=n).astype(dtype))
        if n:
            source[::5] = source[0]
            source.sort()
        queries = np.concatenate((rng.integers(-30, 30, size=79).astype(dtype), np.array([-100, 0, 100], dtype=dtype)))
        if np.issubdtype(dtype, np.floating):
            source = np.sort(np.concatenate((source, np.array([-np.inf, np.inf, np.nan], dtype=dtype))))
            queries = np.concatenate((queries, np.array([-np.inf, np.inf, np.nan, -0.0, 0.0], dtype=dtype)))
        for a in (source, source[::2]):
            for keys in (queries, queries[::-1], queries[::3]):
                for side in ("left", "right"):
                    expected = np.array([sum((v < k) if side == "left" else (v <= k) for v in a) for k in keys], dtype=np.intp)
                    # NumPy comparator considers NaNs sorted after other values:
                    if np.issubdtype(dtype, np.floating):
                        expected = np.array([sum(
                            ((not np.isnan(v)) if np.isnan(k) else
                             (False if np.isnan(v) else v < k))
                            if side == "left" else
                            (True if np.isnan(k) else
                             (False if np.isnan(v) else v <= k))
                            for v in a) for k in keys], dtype=np.intp)
                    actual = np.searchsorted(a, keys, side=side)
                    np.testing.assert_array_equal(actual, expected)
                    tested += 1
print(f"PASS correctness cases={tested} numpy={np.__version__}")
