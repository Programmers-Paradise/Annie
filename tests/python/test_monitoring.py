import socket
import urllib.request

import numpy as np

from rust_annie import AnnIndex, PyMetricsCollector


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_metrics_disabled_by_default():
    assert AnnIndex(2).get_metrics() is None


def test_metrics_count_queries():
    idx = AnnIndex(4)
    idx.enable_metrics()
    idx.add(np.eye(4), np.arange(4))
    for _ in range(5):
        idx.search(np.ones(4), k=2)
    idx.search_batch(np.ones((3, 4)), k=2)
    idx.update_recall_estimate(10, 0.95)
    m = idx.get_metrics()
    assert m["query_count"] == 8
    assert m["index_size"] == 4 and m["dimensions"] == 4 and m["distance_metric"] == "euclidean"
    assert m["recall_estimates"] == {10: 0.95}


def test_prometheus_endpoint():
    port = free_port()
    idx = AnnIndex(2)
    idx.enable_metrics(port)
    idx.add(np.eye(2), [0, 1])
    idx.search([0.0, 0.0], k=1)
    body = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5).read().decode()
    assert "annie_queries_total 1" in body and "annie_index_size 2" in body
    health = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5).read()
    assert health == b"OK"


def test_standalone_collector():
    c = PyMetricsCollector()
    assert c.get_metrics()["query_count"] == 0
    assert "annie_queries_total 0" in c.get_prometheus_metrics()
