"""Restrict results with filters, remove items, and save/load an index."""

import tempfile
from pathlib import Path

import numpy as np

from rust_annie import Distance, Filter, PyHnswIndex

rng = np.random.default_rng(2)
embeddings = rng.standard_normal((5_000, 32)).astype(np.float32)
ids = np.arange(5_000)
year = 2000 + ids % 25  # per-item metadata kept by the caller

index = PyHnswIndex(32, metric=Distance.COSINE)
index.add(embeddings, ids)

query = embeddings[0]
recent = Filter.id_set(ids[year >= 2020].tolist())
found, dists = index.search(query, k=5, filter=recent)
assert all(year[found] >= 2020)
print("recent items:", found)

found, _ = index.search(query, k=5, filter=lambda i: i % 2 == 0)
print("even ids:", found)

index.remove([int(found[0])])

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "embeddings.annie"
    index.save(str(path))
    restored = PyHnswIndex.load(str(path))
    print(f"restored {len(restored)} vectors, metric={restored.metric.name()}")
