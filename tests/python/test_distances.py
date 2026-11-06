import math

import numpy as np
import pytest

from rust_annie import AnnIndex, Distance

from _helpers import exact_knn

A = np.array([0.5, -1.0, 2.0, 0.0], dtype=np.float32)
B = np.array([1.5, 1.0, -0.5, 3.0], dtype=np.float32)
BIN_A = np.array([1, 0, 1, 1], dtype=np.float32)
BIN_B = np.array([1, 1, 0, 1], dtype=np.float32)


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 1.0 if na == 0 or nb == 0 else 1 - np.dot(a, b) / (na * nb)


def angular(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return math.pi / 2
    return math.acos(np.clip(np.dot(a, b) / (na * nb), -1, 1))


def jaccard(a, b):
    a, b = a > 0.5, b > 0.5
    union = np.sum(a | b)
    return 0.0 if union == 0 else 1 - np.sum(a & b) / union


def canberra(a, b):
    den = np.abs(a) + np.abs(b)
    return np.sum(np.where(den > 0, np.abs(a - b) / np.where(den > 0, den, 1), 0))


REFERENCE = [
    (Distance.EUCLIDEAN, lambda a, b: np.linalg.norm(a - b), False),
    (Distance.COSINE, cosine, False),
    (Distance.MANHATTAN, lambda a, b: np.sum(np.abs(a - b)), False),
    (Distance.CHEBYSHEV, lambda a, b: np.max(np.abs(a - b)), False),
    (Distance.ANGULAR, angular, False),
    (Distance.CANBERRA, canberra, False),
    (Distance.Minkowski(3.0), lambda a, b: np.linalg.norm(a - b, ord=3), False),
    (Distance.HAMMING, lambda a, b: float(np.sum(a != b)), True),
    (Distance.JACCARD, jaccard, True),
]


@pytest.mark.parametrize(
    "metric, ref, binary", REFERENCE, ids=lambda x: repr(x) if isinstance(x, Distance) else ""
)
def test_pairwise_distance(metric, ref, binary):
    a, b = (BIN_A, BIN_B) if binary else (A, B)
    idx = AnnIndex(4, metric)
    idx.add(a[None, :], [0])
    _, d = idx.search(b, k=1)
    np.testing.assert_allclose(d[0], ref(a, b), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "metric, ref, binary",
    [r for r in REFERENCE if not r[2]],
    ids=lambda x: repr(x) if isinstance(x, Distance) else "",
)
def test_ranking_matches_reference(metric, ref, binary, dataset):
    data, ids, queries = dataset
    idx = AnnIndex(16, metric)
    idx.add(data, ids)
    for q in queries[:5]:
        got_ids, got_d = idx.search(q, k=10)
        want_ids, want_d = exact_knn(data, ids, q, 10, ref)
        np.testing.assert_allclose(got_d, want_d, rtol=1e-4, atol=1e-5)
        np.testing.assert_array_equal(got_ids, want_ids)


@pytest.mark.parametrize("p", [1.0, 1.5, 2.0, 4.5])
def test_minkowski_constructors_agree(p):
    a = AnnIndex.new_minkowski(4, p)
    b = AnnIndex(4, Distance.Minkowski(p))
    for idx in (a, b):
        idx.add(A[None, :], [0])
        np.testing.assert_allclose(idx.search(B, k=1)[1][0], np.linalg.norm(A - B, ord=p), rtol=1e-5)


@pytest.mark.parametrize("p", [0.0, -1.0, float("nan")])
def test_minkowski_invalid_p(p):
    with pytest.raises(ValueError):
        AnnIndex.new_minkowski(4, p)


def test_cosine_zero_vector():
    idx = AnnIndex(2, Distance.COSINE)
    idx.add(np.zeros((1, 2)), [0])
    assert idx.search([1.0, 0.0], k=1)[1][0] == pytest.approx(1.0)


def test_distance_equality_and_names():
    assert Distance.EUCLIDEAN == Distance.EUCLIDEAN
    assert Distance.EUCLIDEAN != Distance.COSINE
    assert Distance("Cosine") == Distance.COSINE
    assert Distance.Minkowski(3.0) == Distance.Minkowski(3.0)
    assert Distance.custom("x").name() == "x"
    assert Distance.MANHATTAN.name() == "manhattan"
    assert repr(Distance.EUCLIDEAN) == "Distance.EUCLIDEAN"
