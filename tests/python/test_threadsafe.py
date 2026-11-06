import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from rust_annie import AnnIndex, Distance, ThreadSafeAnnIndex, register_metric


def test_same_results_as_ann_index(dataset):
    data, ids, queries = dataset
    a, t = AnnIndex(16), ThreadSafeAnnIndex(16)
    a.add(data, ids)
    t.add(data, ids)
    np.testing.assert_array_equal(a.search_batch(queries, k=5)[0], t.search_batch(queries, k=5)[0])
    assert len(t) == 500 and t.dim() == 16 and 1000 in t


def test_concurrent_readers_and_writers(rng):
    idx = ThreadSafeAnnIndex(8)
    idx.add(rng.standard_normal((100, 8)), np.arange(100))
    errors = []

    def writer(w):
        try:
            for j in range(20):
                base = 1000 + w * 100 + j * 2
                idx.add(np.full((2, 8), base, dtype=np.float32), [base, base + 1])
                idx.remove([base])
        except Exception as e:  # pragma: no cover
            errors.append(e)

    def reader():
        try:
            for _ in range(50):
                ids, d = idx.search(np.zeros(8), k=5)
                assert len(ids) == 5 and np.all(np.diff(d) >= 0)
                idx.search_batch(np.zeros((4, 8)), k=3, filter=lambda i: i < 1000)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(w,)) for w in range(4)]
    threads += [threading.Thread(target=reader) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in threads), "deadlock"
    assert not errors, errors
    assert len(idx) == 100 + 4 * 20


def test_custom_metric_from_threads():
    register_metric("test_threaded_l1", lambda a, b: float(np.abs(a - b).sum()))
    idx = ThreadSafeAnnIndex(2, Distance.custom("test_threaded_l1"))
    idx.add(np.eye(2), [0, 1])
    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(lambda _: idx.search([1.0, 0.0], k=1)[0][0], range(32)))
    assert results == [0] * 32


def test_errors_propagate():
    idx = ThreadSafeAnnIndex(2)
    with pytest.raises(KeyError):
        idx.remove([1])
    with pytest.raises(ValueError):
        idx.add(np.zeros((1, 3)), [0])
