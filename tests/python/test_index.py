import numpy as np
import pytest

from rust_annie import AnnIndex, Distance, Index


@pytest.mark.parametrize("kind", ["brute", "hnsw", "HNSW"])
def test_add_search_remove(kind, dataset):
    data, ids, queries = dataset
    idx = Index(kind, 16, Distance.EUCLIDEAN)
    idx.add(data, ids)
    assert len(idx) == 500 and idx.kind == kind.lower()
    exact = AnnIndex(16)
    exact.add(data, ids)
    got_ids, got_d = idx.search(queries[0], k=5)
    want_ids, want_d = exact.search(queries[0], k=5)
    assert len(set(got_ids) & set(want_ids)) >= 4
    assert np.all(got_d > 0), "distances must be real, not placeholders"
    b_ids, _ = idx.search_batch(queries, k=5)
    assert b_ids.shape == (20, 5)
    idx.remove([int(got_ids[0])])
    assert len(idx) == 499
    idx.validate()
    assert idx.get_info()["type"] == kind.lower()


def test_invalid_backend():
    with pytest.raises(ValueError, match="Unknown backend"):
        Index("faiss", 4)
