import numpy as np
from rust_annie import AnnIndex, Distance

index = AnnIndex.new(3, Distance.Euclidean)
data = np.array([
    [0.1, 0.2, 0.3],
    [1.0, 1.0, 1.0],
    [2.0, 2.0, 2.0],
    [3.0, 3.0, 3.0],
], dtype=np.float32)
ids = np.array([1, 2, 3, 4], dtype=np.int64)

index.add(data, ids)

def even_id_filter(i):
    return i % 2 == 0

query = np.array([0.0, 0.0, 0.0], dtype=np.float32)
allowed_ids = set(filter(even_id_filter, ids))
result = index.search_filter(query, k=2, allowed_ids=allowed_ids)
ids, dists = result[0], result[1]
print("Filtered IDs:", ids)
print("Distances:", dists)
