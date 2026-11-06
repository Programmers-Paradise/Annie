"""Fast exact and approximate nearest-neighbour search, implemented in Rust.

Main classes
------------
AnnIndex
    Exact (brute-force) k-NN search with many distance metrics, custom
    Python metrics, filtering, removal and persistence.
ThreadSafeAnnIndex
    ``AnnIndex`` that can be shared between Python threads.
PyHnswIndex
    Approximate k-NN search using HNSW graphs.
Index
    One class with a selectable ``"brute"`` or ``"hnsw"`` backend.
"""

from ._rust_annie import (
    AnnIndex,
    Distance,
    Filter,
    HnswConfig,
    Index,
    PyHnswIndex,
    PyMetricsCollector,
    ThreadSafeAnnIndex,
    __version__,
    list_metrics,
    register_metric,
)

__all__ = [
    "AnnIndex",
    "Distance",
    "Filter",
    "HnswConfig",
    "Index",
    "PyHnswIndex",
    "PyMetricsCollector",
    "ThreadSafeAnnIndex",
    "__version__",
    "list_metrics",
    "register_metric",
]
