"""Exact search under a Mahalanobis distance supplied as a Python function.

Custom metrics run one Python call per comparison, so they suit small to
medium collections where the metric matters more than raw speed.
"""

import numpy as np

from rust_annie import AnnIndex, register_metric

rng = np.random.default_rng(1)
cov = np.array([[4.0, 1.5], [1.5, 1.0]])
samples = rng.multivariate_normal([0, 0], cov, size=500).astype(np.float32)
cov_inv = np.linalg.inv(np.cov(samples.T))


def mahalanobis(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float64) - b
    return float(np.sqrt(d @ cov_inv @ d))


register_metric("mahalanobis", mahalanobis)
index = AnnIndex.new_with_metric(2, "mahalanobis")
index.add(samples, np.arange(len(samples)))

ids, dists = index.search(np.array([3.0, 0.0]), k=5)
print("nearest under Mahalanobis:", ids, np.round(dists, 3))
