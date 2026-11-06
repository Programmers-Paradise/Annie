"""Reference implementations used as ground truth in tests."""

import numpy as np


def exact_knn(data, ids, query, k, dist_fn):
    d = np.array([dist_fn(query, v) for v in data])
    order = np.lexsort((ids, d))[:k]
    return ids[order], d[order]


def euclidean(a, b):
    return float(np.sqrt(np.sum((a.astype(np.float64) - b) ** 2)))


def recall(found, truth):
    return len(set(found.tolist()) & set(truth.tolist())) / len(truth)
