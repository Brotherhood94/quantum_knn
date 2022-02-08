import numpy as np
from collections import defaultdict


class LSH:
    def __init__(self, n_features, n_bits, num_tables=10, random_seed=None):
        self.hash_size = n_bits
        self.inp_dimensions = n_features
        self.num_tables = num_tables
        self.projections_list = list()
        self.random_seed = random_seed
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        for i in range(self.num_tables):
            self.projections_list.append(np.random.randn(self.hash_size, n_features))

    def _get_hash_encoding(self, x, projections):
        bools = (np.dot(x, projections.T) > 0).astype('int')
        return ''.join(bools.astype('str'))

    def get_hash_encoding(self, x):
        results = defaultdict(int)
        for projections in self.projections_list:
            h = self._get_hash_encoding(x, projections)
            results[h] += 1
        h = max(results, key=results.get)
        return h
