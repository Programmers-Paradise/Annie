"""Property-based tests: exact search must agree with a NumPy reference."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from rust_annie import AnnIndex, Distance

finite = st.floats(-100, 100, allow_nan=False, width=32)


@st.composite
def problems(draw):
    dim = draw(st.integers(1, 8))
    n = draw(st.integers(1, 40))
    data = draw(arrays(np.float32, (n, dim), elements=finite))
    q = draw(arrays(np.float32, (dim,), elements=finite))
    k = draw(st.integers(1, n + 3))
    return data, q, k


@settings(max_examples=150, deadline=None)
@given(problems())
def test_euclidean_matches_numpy(problem):
    data, q, k = problem
    idx = AnnIndex(data.shape[1], Distance.EUCLIDEAN)
    idx.add(data, np.arange(len(data)))
    ids, d = idx.search(q, k=k)
    ref = np.sqrt(((data.astype(np.float64) - q) ** 2).sum(axis=1))
    assert len(ids) == min(k, len(data))
    np.testing.assert_allclose(d, np.sort(ref)[: len(ids)], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(ref[ids], d, rtol=1e-4, atol=1e-3)


@settings(max_examples=100, deadline=None)
@given(problems())
def test_manhattan_batch_equals_single(problem):
    data, q, k = problem
    idx = AnnIndex(data.shape[1], Distance.MANHATTAN)
    idx.add(data, np.arange(len(data)))
    single = idx.search(q, k=k)
    b_ids, b_d = idx.search_batch(q[None, :], k=k)
    m = len(single[0])
    np.testing.assert_array_equal(b_ids[0, :m], single[0])
    assert (b_ids[0, m:] == -1).all()
