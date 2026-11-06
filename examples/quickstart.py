"""Exact and approximate search on the same data, and how to compare them."""

import numpy as np

from rust_annie import AnnIndex, Distance, PyHnswIndex

rng = np.random.default_rng(0)
data = rng.standard_normal((20_000, 64)).astype(np.float32)
ids = np.arange(len(data))
queries = rng.standard_normal((100, 64)).astype(np.float32)

exact = AnnIndex(64, Distance.EUCLIDEAN)
exact.add(data, ids)
true_ids, _ = exact.search_batch(queries, k=10)

approx = PyHnswIndex(64, metric=Distance.EUCLIDEAN, m=16, ef_construction=200)
approx.add(data, ids)

for ef in (10, 50, 200):
    found, _ = approx.search_batch(queries, k=10, ef=ef)
    recall = np.mean([len(set(f) & set(t)) / 10 for f, t in zip(found, true_ids, strict=True)])
    print(f"HNSW ef={ef:<4d} recall@10 = {recall:.3f}")
