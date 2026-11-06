import numpy as np
import pytest

from rust_annie import AnnIndex, Distance, Index, PyHnswIndex, ThreadSafeAnnIndex


def test_ann_index_roundtrip(tmp_path, dataset):
    data, ids, queries = dataset
    idx = AnnIndex(16, Distance.COSINE)
    idx.add(data, ids)
    idx.set_max_deleted_ratio(0.9)
    idx.remove([1000, 1001])
    path = tmp_path / "brute.annie"
    idx.save(str(path))
    assert path.exists(), "save must write exactly the given path"
    loaded = AnnIndex.load(str(path))
    assert len(loaded) == 498 and loaded.metric == Distance.COSINE and 1000 not in loaded
    for q in queries[:5]:
        a, b = idx.search(q, k=5), loaded.search(q, k=5)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_allclose(a[1], b[1])
    loaded.validate()


def test_minkowski_metric_persisted(tmp_path):
    idx = AnnIndex.new_minkowski(2, 3.0)
    idx.add(np.eye(2), [0, 1])
    idx.save(str(tmp_path / "m.bin"))
    assert AnnIndex.load(str(tmp_path / "m.bin")).metric == Distance.Minkowski(3.0)


def test_hnsw_roundtrip(tmp_path, dataset):
    data, ids, queries = dataset
    idx = PyHnswIndex(16, metric=Distance.COSINE, m=12)
    idx.add(data, ids)
    idx.remove([1000])
    idx.save(str(tmp_path / "h.annie"))
    loaded = PyHnswIndex.load(str(tmp_path / "h.annie"))
    assert len(loaded) == 499 and 1000 not in loaded
    assert loaded.metric == Distance.COSINE and loaded.config.m == 12
    exact = AnnIndex(16, Distance.COSINE)
    exact.add(data[1:], ids[1:])
    for q in queries[:5]:
        overlap = set(loaded.search(q, k=10)[0]) & set(exact.search(q, k=10)[0])
        assert len(overlap) >= 8


def test_threadsafe_roundtrip(tmp_path):
    idx = ThreadSafeAnnIndex(2)
    idx.add(np.eye(2), [5, 6])
    idx.save(str(tmp_path / "t.bin"))
    assert len(ThreadSafeAnnIndex.load(str(tmp_path / "t.bin"))) == 2


def test_index_load_detects_kind(tmp_path):
    for kind in ("brute", "hnsw"):
        idx = Index(kind, 2)
        idx.add(np.eye(2), [0, 1])
        p = str(tmp_path / f"{kind}.bin")
        idx.save(p)
        assert Index.load(p).kind == kind


def test_wrong_kind_rejected(tmp_path):
    h = PyHnswIndex(2)
    h.save(str(tmp_path / "h.bin"))
    with pytest.raises(OSError, match="Hnsw"):
        AnnIndex.load(str(tmp_path / "h.bin"))


def test_load_errors(tmp_path):
    with pytest.raises(OSError):
        AnnIndex.load(str(tmp_path / "missing.bin"))
    junk = tmp_path / "junk.bin"
    junk.write_bytes(b"not an index at all")
    with pytest.raises(OSError, match="not a rust_annie index"):
        AnnIndex.load(str(junk))
    with pytest.raises(OSError):
        AnnIndex(2).save(str(tmp_path))  # a directory
    with pytest.raises(OSError):
        AnnIndex(2).save("bad\0name")
