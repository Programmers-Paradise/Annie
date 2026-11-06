import numpy as np
import pytest

from rust_annie import AnnIndex, Filter, Index, PyHnswIndex, ThreadSafeAnnIndex


@pytest.fixture
def line_index():
    idx = AnnIndex(1)
    idx.add(np.arange(10, dtype=np.float32)[:, None], np.arange(10))
    return idx


def found(idx, flt, k=10):
    return sorted(idx.search([0.0], k=k, filter=flt)[0].tolist())


def test_id_range(line_index):
    assert found(line_index, Filter.id_range(3, 5)) == [3, 4, 5]


def test_id_set(line_index):
    assert found(line_index, Filter.id_set([1, 8, 42])) == [1, 8]


def test_boolean(line_index):
    bits = [i % 3 == 0 for i in range(10)]
    assert found(line_index, Filter.boolean(bits)) == [0, 3, 6, 9]


def test_combinators(line_index):
    f = Filter.and_([Filter.id_range(0, 6), Filter.not_(Filter.id_set([2, 4]))])
    assert found(line_index, f) == [0, 1, 3, 5, 6]
    g = Filter.or_([Filter.id_set([1]), Filter.id_range(8, 9)])
    assert found(line_index, g) == [1, 8, 9]


def test_callable(line_index):
    assert found(line_index, lambda i: i % 2 == 1) == [1, 3, 5, 7, 9]


def test_filter_returns_nearest_matches(line_index):
    ids, d = line_index.search([0.0], k=2, filter=lambda i: i >= 5)
    assert ids.tolist() == [5, 6] and d.tolist() == [5.0, 6.0]


def test_batch_with_filter_pads(line_index):
    ids, d = line_index.search_batch(np.zeros((2, 1)), k=3, filter=Filter.id_set([7]))
    assert ids.tolist() == [[7, -1, -1], [7, -1, -1]]
    assert np.isinf(d[:, 1:]).all()


def test_callable_exception_propagates(line_index):
    def bad(i):
        raise KeyError("nope")

    with pytest.raises(KeyError):
        line_index.search([0.0], k=1, filter=bad)


def test_invalid_filter_type(line_index):
    with pytest.raises(TypeError):
        line_index.search([0.0], k=1, filter=42)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ThreadSafeAnnIndex(1),
        lambda: PyHnswIndex(1),
        lambda: Index("brute", 1),
        lambda: Index("hnsw", 1),
    ],
)
def test_filters_on_every_index_type(factory):
    idx = factory()
    idx.add(np.arange(50, dtype=np.float32)[:, None], np.arange(50))
    ids, _ = idx.search([0.0], k=3, filter=lambda i: i % 10 == 0)
    assert ids.tolist() == [0, 10, 20]
    ids, _ = idx.search([0.0], k=3, filter=Filter.id_range(40, 49))
    assert ids.tolist() == [40, 41, 42]
