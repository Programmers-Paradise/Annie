# API reference

Everything is importable from the top-level `rust_annie` package. Type stubs are
shipped with the package, so IDEs and type checkers see these signatures.

Conventions used below: *array-like* means anything NumPy can convert (arrays of any
float/int dtype, lists); vectors are converted to `float32` and IDs to `int64`. A
*filter* is `None`, a [`Filter`](#filter), or a callable `f(id) -> bool`.

## `AnnIndex`

Exact k-nearest-neighbour search.

```text
AnnIndex(dim: int, metric: Distance = Distance.EUCLIDEAN)
AnnIndex.new_minkowski(dim: int, p: float) -> AnnIndex
AnnIndex.new_with_metric(dim: int, metric_name: str) -> AnnIndex
```

`dim` must be between 1 and 65 536. `new_with_metric` accepts built-in names
(`"euclidean"`, `"cosine"`, …) and names registered with
[`register_metric`](#register_metric); unknown names raise `ValueError`.

| Method | Description |
|---|---|
| `add(data, ids)` | Add an `(n, dim)` array-like with `n` unique non-negative IDs. Atomic: on error nothing is added. |
| `add_batch_with_progress(data, ids, progress_callback, chunk_size=10000)` | As `add`, calling `progress_callback(done, total)` after each chunk. |
| `search(query, k, filter=None) -> (ids, distances)` | `k` nearest neighbours of one `(dim,)` query, sorted by distance. Returns `int64` and `float32` arrays of length ≤ `k`. |
| `search_batch(queries, k, filter=None) -> (ids, distances)` | Arrays of shape `(n, k)`; missing results padded with `-1` / `inf`. |
| `remove(ids)` | Remove vectors. `KeyError` if any ID is absent (nothing removed). |
| `update(id, vector)` | Replace the vector for `id`. `KeyError` if absent. |
| `compact()` | Reclaim storage of removed vectors now. |
| `set_max_deleted_ratio(ratio)` | Fraction (0–1) of removed slots that triggers automatic compaction. Default 0.2. |
| `save(path)` / `AnnIndex.load(path)` | Write / read a binary index file at exactly `path`. |
| `ids() -> ndarray` | IDs of all stored vectors. |
| `len(index)`, `index.len()` | Number of stored vectors. |
| `id in index` | Whether an ID is stored. |
| `dim()` | Dimensionality. |
| `metric` | The `Distance` in use (property). |
| `capacity()`, `deleted_count()` | Storage slots in use, removed-but-not-compacted slots. |
| `version()` | Counter incremented on every mutation. |
| `enable_metrics(port=None)` | Start recording query metrics; with `port`, serve Prometheus metrics on 127.0.0.1. |
| `get_metrics() -> dict \| None` | `query_count`, `avg_query_latency_us`, `index_size`, `dimensions`, `uptime_seconds`, `distance_metric`, `recall_estimates`. |
| `update_recall_estimate(k, recall)` | Record an externally measured recall@k. |
| `get_info() -> dict[str, str]` | Size, metric, memory estimate and other details. |
| `validate()` | Check internal consistency; `ValueError` on problems. |

## `PyHnswIndex`

Approximate search with a hierarchical navigable small-world graph.

```text
PyHnswIndex(dims: int, metric: Distance = Distance.EUCLIDEAN, m: int = 16,
            ef_construction: int = 200, ef_search: int = 50,
            max_elements: int = 10000, config: HnswConfig | None = None)
```

`metric` must be `EUCLIDEAN`, `COSINE` or `MANHATTAN`. `m` must be in 2–256. If `config`
is given it overrides the four individual parameters.

| Method | Description |
|---|---|
| `add(data, ids)` | Add vectors (built in parallel). Atomic validation as for `AnnIndex`. |
| `search(query, k, ef=None, filter=None)` | Approximate neighbours; `ef` (default `ef_search`, at least `k`) trades speed for recall. |
| `search_batch(queries, k, ef=None, filter=None)` | Batch version, parallel across queries; `(n, k)` arrays padded with `-1` / `inf`. |
| `remove(ids)` | Exclude vectors from results. `KeyError` if any ID is absent. |
| `save(path)` / `PyHnswIndex.load(path)` | Persist vectors and parameters; the graph is rebuilt on load. |
| `ef_search` | Default search `ef` (read/write property). |
| `metric`, `config` | Properties. |
| `dims()`, `len()`, `len(index)`, `id in index`, `get_info()`, `validate()` | As for `AnnIndex`. |

## `HnswConfig`

```text
HnswConfig(m=16, ef_construction=200, ef_search=50, max_elements=10000)
HnswConfig.default()
```

Mutable attributes `m`, `ef_construction`, `ef_search`, `max_elements`;
`validate()` raises `ValueError` for out-of-range values.

## `ThreadSafeAnnIndex`

```text
ThreadSafeAnnIndex(dim: int, metric: Distance = Distance.EUCLIDEAN)
```

Same behaviour as `AnnIndex`, safe to share between threads (a readers–writer lock;
searches run concurrently, mutations are exclusive). Methods: `add`, `search`,
`search_batch`, `remove`, `update`, `compact`, `save`, `load` (static), `len`, `dim`,
`version`, `len(index)`, `id in index`.

## `Index`

```text
Index(kind: str, dim: int, metric: Distance = Distance.EUCLIDEAN)
```

`kind` is `"brute"` (an `AnnIndex`) or `"hnsw"` (a `PyHnswIndex` with default parameters).
Methods: `add`, `search`, `search_batch`, `remove`, `save`, `Index.load(path)` (detects the
kind from the file), `len(index)`, `get_info`, `validate`; property `kind`.

## `Distance`

Class attributes `EUCLIDEAN`, `COSINE`, `ANGULAR`, `MANHATTAN`, `CHEBYSHEV`, `CANBERRA`,
`HAMMING`, `JACCARD`; constructors `Distance.Minkowski(p)`, `Distance.custom(name)`, and
`Distance(name)` (by name, case-insensitive; unknown names give a custom metric).
`name()` returns the canonical name. Distances compare equal by value and are hashable.
Definitions are in the [user guide](user-guide.md#distance-metrics).

## `Filter`

| Constructor | Accepts |
|---|---|
| `Filter.id_range(min, max)` | IDs with `min <= id <= max` |
| `Filter.id_set(ids)` | IDs in `ids` |
| `Filter.boolean(bits)` | entries at storage positions `i` with `bits[i]` true |
| `Filter.and_(filters)` | IDs accepted by all `filters` |
| `Filter.or_(filters)` | IDs accepted by any of `filters` |
| `Filter.not_(filter)` | IDs rejected by `filter` |

`accepts(id, position) -> bool` evaluates the filter directly.

## `register_metric`

```text
register_metric(name: str, func: Callable[[ndarray, ndarray], float]) -> None
list_metrics() -> list[str]
```

Registers `func` under `name` for use with `AnnIndex.new_with_metric(dim, name)` or
`Distance.custom(name)`. Built-in names cannot be overridden. Registration is per
process.

## `PyMetricsCollector`

A stand-alone metrics collector with `enable_metrics(port)`, `get_metrics()` and
`get_prometheus_metrics()`, for applications that record their own query statistics.

## Exceptions

| Situation | Exception |
|---|---|
| Wrong shape, dimension, NaN/inf input, duplicate ID, invalid parameter, unknown metric | `ValueError` |
| Unknown ID in `remove` / `update` | `KeyError` |
| Missing, unreadable, corrupt or wrong-kind index file; invalid path | `OSError` |
| Error raised by a custom metric | `RuntimeError` (message includes the original error) |
| Error raised by a filter callable | propagated unchanged |
| Filter of the wrong type | `TypeError` |
