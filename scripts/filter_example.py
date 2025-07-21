import numpy as np
from rust_annie import AnnIndex, Distance
from typing import Set, Tuple

index = AnnIndex.new(3, Distance.Euclidean)
data = np.array([
    [0.1, 0.2, 0.3],
    [1.0, 1.0, 1.0],
    [2.0, 2.0, 2.0],
    [3.0, 3.0, 3.0],
], dtype=np.float32)
ids = np.array([1, 2, 3, 4], dtype=np.int64)

index.add(data, ids)

def even_id_filter(i: int) -> bool:
    """
    Filter function to keep only even numbered IDs.

    Args:
        i (int): An ID.

    Returns:
        bool: True if ID is even, False otherwise.
    """
    return i % 2 == 0

query = np.array([0.0, 0.0, 0.0], dtype=np.float32)
allowed_ids: Set[int] = set(filter(even_id_filter, ids))
result: Tuple[np.ndarray, np.ndarray] = index.search_filter(query, k=2, allowed_ids=list(allowed_ids))
ids, dists = result[0], result[1]

print("Filtered IDs:", ids)
print("Distances:", dists)
