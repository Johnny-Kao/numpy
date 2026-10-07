from pathlib import Path
import sys

TARGET = Path("numpy/_core/tests/test_multiarray.py")
MARKER = "    def test_searchsorted_unicode(self):\n"
SENTINEL = "    def test_searchsorted_locality_large_ordered_numeric(self):\n"

BLOCK = r'''    def test_searchsorted_locality_large_ordered_numeric(self):
        # Exercise the locality finisher above its 2**20-key activation gate.
        # These representative dtypes cover signed/unsigned integers plus the
        # float16/float32/float64 type-specific search implementations.
        key_count = 1 << 20
        signed_values = [-1, 0, 1, 31, 32, 63, 64, 65]
        signed_expected = {
            'left': [0, 0, 4, 124, 128, 252, 256, 256],
            'right': [0, 4, 8, 128, 132, 256, 256, 256],
        }
        unsigned_values = [0, 1, 31, 32, 63, 64, 65, 255]
        unsigned_expected = {
            'left': [0, 4, 124, 128, 252, 256, 256, 256],
            'right': [4, 8, 128, 132, 256, 256, 256, 256],
        }

        for dtype in [np.int32, np.int64, np.uint64,
                      np.float16, np.float32, np.float64]:
            a = np.repeat(np.arange(64, dtype=dtype), 4)
            if np.issubdtype(dtype, np.unsignedinteger):
                values = unsigned_values
                expected = unsigned_expected
            else:
                values = signed_values
                expected = signed_expected
            queries = np.repeat(
                np.asarray(values, dtype=dtype),
                key_count // len(values),
            )
            assert queries.size == key_count
            assert queries.strides == (queries.itemsize,)
            assert a.strides == (a.itemsize,)

            for side in ['left', 'right']:
                actual = a.searchsorted(queries, side=side)
                desired = np.repeat(
                    np.asarray(expected[side], dtype=np.intp),
                    key_count // len(values),
                )
                assert_array_equal(actual, desired)

    def test_searchsorted_locality_large_nan(self):
        # Preserve NumPy floating-point NaN ordering while exercising the
        # accepted locality path.  The query blocks are globally ordered.
        key_count = 1 << 20
        query_values = [-np.inf, 0, 1, 127, 247, np.inf, np.nan, np.nan]
        expected = {
            'left': [0, 0, 1, 127, 247, 248, 250, 250],
            'right': [0, 1, 2, 128, 248, 250, 256, 256],
        }

        for dtype in [np.float16, np.float32, np.float64]:
            a = np.concatenate([
                np.arange(248, dtype=dtype),
                np.asarray([np.inf, np.inf] + [np.nan] * 6, dtype=dtype),
            ])
            queries = np.repeat(
                np.asarray(query_values, dtype=dtype),
                key_count // len(query_values),
            )
            assert queries.size == key_count
            assert queries.strides == (queries.itemsize,)
            assert a.strides == (a.itemsize,)

            for side in ['left', 'right']:
                actual = a.searchsorted(queries, side=side)
                desired = np.repeat(
                    np.asarray(expected[side], dtype=np.intp),
                    key_count // len(query_values),
                )
                assert_array_equal(actual, desired)

    def test_searchsorted_locality_large_complex_nan(self):
        # Complex search uses NumPy's type-specific total ordering, including
        # NaNs.  Repeat the canonical existing ordering with duplicates so the
        # new locality finisher is exercised, not only the historical path.
        key_count = 1 << 20
        counts = np.full(9, key_count // 9, dtype=np.intp)
        counts[-1] += key_count - int(counts.sum())

        for dtype in [np.complex64, np.complex128]:
            values = np.asarray([
                0 + 0j,
                0 + 1j,
                1 + 0j,
                1 + 1j,
                complex(0, np.nan),
                complex(1, np.nan),
                complex(np.nan, 0),
                complex(np.nan, 1),
                complex(np.nan, np.nan),
            ], dtype=dtype)
            a = np.repeat(values, 4)
            queries = np.repeat(values, counts)
            assert queries.size == key_count
            assert queries.strides == (queries.itemsize,)
            assert a.strides == (a.itemsize,)

            for side, expected in [
                ('left', np.arange(0, 36, 4, dtype=np.intp)),
                ('right', np.arange(4, 40, 4, dtype=np.intp)),
            ]:
                actual = a.searchsorted(queries, side=side)
                desired = np.repeat(expected, counts)
                assert_array_equal(actual, desired)

    def test_searchsorted_locality_large_datetime_nat(self):
        # datetime64/timedelta64 have type-specific NaT ordering and are part
        # of the same generated numeric binsearch map as the optimized path.
        key_count = 1 << 20
        cases = [
            ('datetime64[D]', ['2000-01-01', '2000-01-02',
                               '2000-01-03', 'NaT']),
            ('timedelta64[D]', [0, 1, 2, 'NaT']),
        ]
        expected = {
            'left': np.asarray([0, 64, 128, 192], dtype=np.intp),
            'right': np.asarray([64, 128, 192, 256], dtype=np.intp),
        }

        for dtype, values in cases:
            values = np.asarray(values, dtype=dtype)
            a = np.repeat(values, 64)
            queries = np.repeat(values, key_count // len(values))
            assert queries.size == key_count
            assert queries.strides == (queries.itemsize,)
            assert a.strides == (a.itemsize,)

            for side in ['left', 'right']:
                actual = a.searchsorted(queries, side=side)
                desired = np.repeat(
                    expected[side], key_count // len(values)
                )
                assert_array_equal(actual, desired)

    def test_searchsorted_locality_large_disordered(self):
        # Exercise binsearch_locality above the Q gate with a hostile query
        # order whose selector samples produce a decreasing trajectory.  This
        # should take the batched fallback after the first three coarse levels.
        key_count = 1 << 20
        a = np.arange(0, 512, 2, dtype=np.int32)
        pattern = np.asarray([
            511, -1, 256, 3, 510, 128, 0, 255,
            400, 16, 480, 160, 8, 384, 64, 320,
        ], dtype=np.int32)
        queries = np.tile(pattern, key_count // pattern.size)
        assert queries.size == key_count
        assert queries.strides == (queries.itemsize,)
        assert a.strides == (a.itemsize,)

        for side in ['left', 'right']:
            if side == 'left':
                expected_pattern = np.asarray(
                    [(a < value).sum() for value in pattern], dtype=np.intp
                )
            else:
                expected_pattern = np.asarray(
                    [(a <= value).sum() for value in pattern], dtype=np.intp
                )
            desired = np.tile(expected_pattern, key_count // pattern.size)
            actual = a.searchsorted(queries, side=side)
            assert_array_equal(actual, desired)

'''


def main() -> int:
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found; run from the NumPy repository root.", file=sys.stderr)
        return 2

    text = TARGET.read_text(encoding="utf-8")
    if SENTINEL in text:
        print("Tests are already present; no changes made.")
        return 0
    if MARKER not in text:
        print("ERROR: insertion marker not found; inspect the current test file manually.", file=sys.stderr)
        return 3

    TARGET.write_text(text.replace(MARKER, BLOCK + MARKER, 1), encoding="utf-8")
    print(f"Inserted PR #32895 locality tests into {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
