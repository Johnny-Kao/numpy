import numpy as np

from .common import Benchmark


class SearchSorted(Benchmark):
    params = [
        [100, 10_000, 1_000_000, 100_000_000],  # array sizes
        [1, 10, 100_000],                       # number of query elements
        ['ordered', 'random'],                  # query order
        [False, True],                          # use sorter
        [42, 18122022],                         # seed
    ]
    param_names = ['array_size', 'n_queries', 'query_order', 'use_sorter', 'seed']

    def setup(self, array_size, n_queries, query_order, use_sorter, seed):
        self.arr = np.arange(array_size, dtype=np.int32)

        rng = np.random.default_rng(seed)

        low = -array_size // 10
        high = array_size + array_size // 10

        self.queries = rng.integers(low, high, size=n_queries, dtype=np.int32)
        if query_order == 'ordered':
            self.queries.sort()

        if use_sorter:
            rng.shuffle(self.arr)
            self.sorter = self.arr.argsort()
        else:
            self.sorter = None

    def time_searchsorted(self, array_size, n_queries, query_order, use_sorter, seed):
        np.searchsorted(self.arr, self.queries, sorter=self.sorter)


class SearchSortedMonotonicity(Benchmark):
    """Research matrix for query-locality effects in searchsorted."""

    params = [
        [10_000, 1_000_000, 100_000_000],
        [10, 1_000, 100_000],
        ['dense', 'medium', 'sparse', 'clustered', 'mostly_monotonic', 'random'],
        [42, 18122022],
    ]
    param_names = ['array_size', 'n_queries', 'query_shape', 'seed']

    def setup(self, array_size, n_queries, query_shape, seed):
        self.arr = np.arange(array_size, dtype=np.int32)
        rng = np.random.default_rng(seed)

        if query_shape == 'random':
            self.queries = rng.integers(
                -array_size // 10,
                array_size + array_size // 10,
                size=n_queries,
                dtype=np.int32,
            )
            return

        start = array_size // 10

        if query_shape == 'dense':
            steps = np.ones(n_queries, dtype=np.int64)
        elif query_shape == 'medium':
            steps = np.full(n_queries, 10, dtype=np.int64)
        elif query_shape == 'sparse':
            steps = np.full(n_queries, max(100, array_size // max(n_queries * 4, 1)),
                            dtype=np.int64)
        elif query_shape == 'clustered':
            steps = np.ones(n_queries, dtype=np.int64)
            steps[4::5] = 64
        elif query_shape == 'mostly_monotonic':
            steps = np.ones(n_queries, dtype=np.int64)
            positions = start + np.cumsum(steps) - 1
            positions[31::32] -= 16
            if positions.max(initial=0) >= array_size:
                raise NotImplementedError
            self.queries = positions.astype(np.int32)
            return
        else:
            raise ValueError(query_shape)

        positions = start + np.cumsum(steps) - steps[0]
        if positions.max(initial=0) >= array_size:
            raise NotImplementedError
        self.queries = positions.astype(np.int32)

    def time_searchsorted(self, array_size, n_queries, query_shape, seed):
        np.searchsorted(self.arr, self.queries)
