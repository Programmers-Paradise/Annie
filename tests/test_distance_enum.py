import numpy as np
import pytest
from rust_annie import Distance, AnnIndex
import math

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
        (Distance.HAMMING, lambda a, b: np.sum(a != b)),
        (Distance.JACCARD, lambda a, b: 1 - np.sum(a & b) / np.sum(a | b) if np.any(a | b) else 0),
        (Distance.ANGULAR, lambda a, b: math.acos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) if np.linalg.norm(a) * np.linalg.norm(b) > 0 else math.pi/2)),
        (Distance.CANBERRA, lambda a, b: np.sum(np.abs(a - b) / (np.abs(a) + np.abs(b)))),
        (Distance.Minkowski(3), lambda a, b: np.linalg.norm(a - b, ord=3)),
    ]
)
def test_distance_behavior(metric, expected_fn):
    if metric in [Distance.HAMMING, Distance.JACCARD]:
        v0 = np.array([0, 1], dtype=np.float32)
        v1 = np.array([1, 0], dtype=np.float32)
    else:
        v0 = np.array([0.0, 0.0], dtype=np.float32)
        v1 = np.array([1.0, 1.0], dtype=np.float32)

    index = AnnIndex(dim=2, metric=metric)
    index.add(np.array([v0]), np.array([0], dtype=np.int64))

  
    labels, dists = index.search(v1, k=1)
    expected = expected_fn(v0, v1)
    
    assert labels[0] == 0
    if metric == Distance.ANGULAR:
        # Allow some floating point tolerance
        assert pytest.approx(dists[0], abs=1e-5) == expected
    else:
        np.testing.assert_allclose(dists[0], expected, rtol=1e-5)

@pytest.mark.parametrize("p", [1.5, 3.0, 4.5])
def test_minkowski_varying_p(p):
    v0 = np.array([0.0, 0.0], dtype=np.float32)
    v1 = np.array([1.0, 1.0], dtype=np.float32)
    
    index = AnnIndex(dim=2, metric=Distance.Minkowski(p))
    index.add(np.array([v0]), np.array([0], dtype=np.int64))
    
    _, dists = index.search(v1, k=1)
    expected = np.linalg.norm(v0 - v1, ord=p)
    
    np.testing.assert_allclose(dists[0], expected, rtol=1e-5)