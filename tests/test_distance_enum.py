import numpy as np
import pytest
from rust_annie import Distance, AnnIndex

# Sample vectors
v0 = np.array([0.0, 0.0], dtype=np.float32)
v1 = np.array([1.0, 1.0], dtype=np.float32)

@pytest.mark.parametrize(
    "metric, expected_fn",
    [
        (Distance.EUCLIDEAN, lambda a, b: np.linalg.norm(a - b)),
        (Distance.COSINE, lambda a, b: 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))),
        (Distance.MANHATTAN, lambda a, b: np.sum(np.abs(a - b))),
        (Distance.CHEBYSHEV, lambda a, b: np.max(np.abs(a - b))),
    ]
)
def test_distance_behavior(metric, expected_fn):
    
    index = AnnIndex(dim=2, metric=metric)
    index.add(np.array([v0]), np.array([0], dtype=np.int64))

  
    labels, dists = index.search(v1, k=1)
    
    assert labels[0] == 0
    np.testing.assert_allclose(dists[0], expected_fn(v0, v1), rtol=1e-5)
