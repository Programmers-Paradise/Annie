# User guide

## Choosing an index

| Class | Search | Metrics | Use when |
|---|---|---|---|
| `AnnIndex` | exact | all built-in + custom Python metrics | you need exact results, unusual metrics, or have up to ~10⁶ vectors |
| `PyHnswIndex` | approximate (HNSW) | Euclidean, cosine, Manhattan | collections are large and ~95–99 % recall is acceptable |
| `ThreadSafeAnnIndex` | exact | as `AnnIndex` (incl. custom) | several Python threads read and write one index |
| `Index("brute" \| "hnsw", dim)` | either | as above | you want to switch backends with one argument |

All indexes use the same conventions:

- Vectors are passed as array-likes of shape `(n, dim)`; they are converted to
  `float32`. NaN or infinite values are rejected.
- Every vector has a caller-chosen, unique, non-negative integer ID (`int64`).
- `search(query, k)` returns `(ids, distances)` as 1-D NumPy arrays sorted by increasing
  distance, with at most `k` entries (fewer if the index or filter has fewer matches).
- `search_batch(queries, k)` returns arrays of shape `(n_queries, k)`. Rows with fewer
  than `k` matches are padded with ID `-1` and distance `inf`.
- Invalid input raises `ValueError`; unknown IDs in `remove`/`update` raise `KeyError`;
  file problems raise `OSError`.

## Distance metrics

| `Distance` | Definition |
|---|---|
| `EUCLIDEAN` | √Σ(aᵢ − bᵢ)² |
| `COSINE` | 1 − a·b / (‖a‖‖b‖); 1 if either vector is zero |
| `ANGULAR` | arccos(a·b / (‖a‖‖b‖)); π/2 if either vector is zero |
| `MANHATTAN` | Σ\|aᵢ − bᵢ\| |
| `CHEBYSHEV` | max\|aᵢ − bᵢ\| |
| `Minkowski(p)` | (Σ\|aᵢ − bᵢ\|ᵖ)^(1/p), p > 0 |
| `CANBERRA` | Σ\|aᵢ − bᵢ\| / (\|aᵢ\| + \|bᵢ\|), skipping 0/0 terms |
| `HAMMING` | number of coordinates differing by more than 10⁻⁵ |
| `JACCARD` | 1 − \|A∩B\| / \|A∪B\| with A = {i : aᵢ > 0.5}; 0 if both empty |

```python
from rust_annie import AnnIndex, Distance

AnnIndex(16, Distance.MANHATTAN)
AnnIndex(16, Distance.Minkowski(3.0))
AnnIndex.new_minkowski(16, 3.0)        # equivalent
AnnIndex.new_with_metric(16, "cosine") # by name
```

### Custom metrics

`register_metric(name, func)` makes a Python function `func(a, b) -> float` available
to exact search. `a` and `b` are 1-D `float32` NumPy arrays. Exceptions raised by `func`
propagate to the caller of `search`.

```python
import numpy as np
from rust_annie import AnnIndex, Distance, register_metric, list_metrics

def weighted_l1(a, b, w=np.array([1.0, 10.0])):
    return float(np.sum(w * np.abs(a - b)))

register_metric("weighted_l1", weighted_l1)
assert "weighted_l1" in list_metrics()

index = AnnIndex(2, Distance.custom("weighted_l1"))  # or AnnIndex.new_with_metric(2, "weighted_l1")
index.add(np.array([[1.0, 0.0], [0.0, 1.0]]), [0, 1])
ids, dists = index.search([0.0, 0.0], k=1)
assert ids[0] == 0 and dists[0] == 1.0
```

Custom metrics require one Python call per comparison and run on a single thread, so
they are much slower than built-in metrics (roughly 50–100× in our tests, depending on
the function and the number of cores). They suit collections of up to roughly 10⁵
vectors, or prototyping a metric before implementing it natively. Metrics
must be registered again in each new Python process before searching an index that uses
them (including one loaded from disk).

## Filtering

Pass `filter=` to any `search` or `search_batch` call. A filter is either a `Filter`
object or any callable `f(id) -> bool`:

```python
import numpy as np
from rust_annie import AnnIndex, Filter

index = AnnIndex(1)
index.add(np.arange(10.0)[:, None], np.arange(10))

def ids_for(flt):
    return index.search([0.0], k=10, filter=flt)[0].tolist()

assert ids_for(Filter.id_range(3, 5)) == [3, 4, 5]          # inclusive
assert ids_for(Filter.id_set([8, 1])) == [1, 8]
assert ids_for(Filter.and_([Filter.id_range(0, 4), Filter.not_(Filter.id_set([2]))])) == [0, 1, 3, 4]
assert ids_for(Filter.or_([Filter.id_set([9]), Filter.id_range(0, 1)])) == [0, 1, 9]
assert ids_for(lambda i: i % 3 == 0) == [0, 3, 6, 9]
```

`Filter.boolean(bits)` accepts the entry at storage position `i` when `bits[i]` is true.
Positions follow insertion order and change after `compact()`, so prefer ID-based
filters unless you manage positions yourself.

Callables are evaluated once per stored vector before the search starts. For repeated
queries with the same predicate, build a `Filter.id_set` once instead.

With `PyHnswIndex`, filtering happens during graph traversal. Very selective filters
(matching a tiny fraction of the data) can lower recall; increase `ef` or use exact
search on the filtered subset.

## Adding, removing and updating

```python
import numpy as np
from rust_annie import AnnIndex

index = AnnIndex(4)
index.add(np.random.rand(1000, 4), np.arange(1000))
index.remove([10, 11])        # KeyError (and no change) if any ID is missing
index.update(12, np.zeros(4)) # replace a vector in place
assert 10 not in index and len(index) == 998
```

`AnnIndex` removes entries lazily and compacts storage automatically when more than
`max_deleted_ratio` (default 0.2, see `set_max_deleted_ratio`) of the slots are empty;
`compact()` does it immediately. `PyHnswIndex.remove` marks vectors as deleted and
excludes them from results; the graph keeps them as routing nodes until the index is saved
and loaded again. `PyHnswIndex` does not support `update`; remove the old ID and add a
new vector instead.

`add` validates the whole batch (shape, finiteness, duplicate IDs) before inserting
anything, so a failed call leaves the index unchanged.

For long insertions, `AnnIndex.add_batch_with_progress(data, ids, callback)` calls
`callback(done, total)` after each chunk.

## Saving and loading

```python
import numpy as np
from rust_annie import AnnIndex, Index, PyHnswIndex

hnsw = PyHnswIndex(8)
hnsw.add(np.random.rand(100, 8), np.arange(100))
hnsw.save("vectors.hnsw")
hnsw = PyHnswIndex.load("vectors.hnsw")

any_index = Index.load("vectors.hnsw")  # detects the index kind from the file header
assert any_index.kind == "hnsw"
```

Files start with a magic header and a format version, so loading a file of the wrong
kind or from an incompatible version fails with a clear `OSError`. Paths are used
exactly as given. HNSW files store the vectors and parameters; the graph is rebuilt on
load (in parallel), which takes about as long as the original construction.

Files written by rust-annie 0.2.x cannot be read by 0.3; rebuild those indexes from the
original vectors.

## Tuning HNSW

| Parameter | Default | Effect |
|---|---|---|
| `m` | 16 | neighbours per node; higher → better recall, more memory, slower build. 12–48 is typical. |
| `ef_construction` | 200 | candidate list size while building; higher → better graph, slower build. |
| `ef_search` / `ef` | 50 | candidate list size while searching; higher → better recall, slower queries. Always at least `k`. |
| `max_elements` | 10 000 | expected size, used only to pre-size internal tables. Not a limit. |

`ef` is the main knob: set it per call with `search(q, k, ef=...)`, or change the default
with `index.ef_search = ...`. To choose it, measure recall against exact search on a
sample of queries:

```python
import numpy as np
from rust_annie import AnnIndex, PyHnswIndex

rng = np.random.default_rng(0)
data = rng.standard_normal((5000, 32)).astype(np.float32)
queries = rng.standard_normal((50, 32)).astype(np.float32)

exact = AnnIndex(32)
exact.add(data, np.arange(5000))
truth, _ = exact.search_batch(queries, k=10)

hnsw = PyHnswIndex(32, m=16, ef_construction=200)
hnsw.add(data, np.arange(5000))
for ef in (16, 64, 256):
    found, _ = hnsw.search_batch(queries, k=10, ef=ef)
    recall = np.mean([len(set(f) & set(t)) / 10 for f, t in zip(found, truth)])
    print(f"ef={ef}: recall@10={recall:.3f}")
```

## Threads and the GIL

All index types release the GIL while searching with built-in metrics, and batch
searches run in parallel across cores (Rayon). Set `RAYON_NUM_THREADS` to limit the
number of worker threads.

`AnnIndex` and `PyHnswIndex` may be searched from several Python threads at once, but
must not be modified while another thread uses them (Python raises
`RuntimeError: Already borrowed`). Use `ThreadSafeAnnIndex` when reads and writes are
interleaved across threads: searches proceed concurrently and mutations take an exclusive
lock.

```python
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from rust_annie import ThreadSafeAnnIndex

index = ThreadSafeAnnIndex(16)
index.add(np.random.rand(10_000, 16), np.arange(10_000))
with ThreadPoolExecutor(8) as pool:
    results = list(pool.map(lambda q: index.search(q, k=5), np.random.rand(100, 16)))
```

## Monitoring

`AnnIndex.enable_metrics()` records query counts and latency. With a port, the metrics are
also served in Prometheus text format on `http://127.0.0.1:<port>/metrics` (and a
`/health` endpoint).

```python
import numpy as np
from rust_annie import AnnIndex

index = AnnIndex(4)
index.enable_metrics()  # or enable_metrics(port=9100)
index.add(np.random.rand(100, 4), np.arange(100))
index.search(np.zeros(4), k=3)
index.update_recall_estimate(10, 0.97)  # record a recall you measured yourself
stats = index.get_metrics()
assert stats["query_count"] == 1 and stats["recall_estimates"] == {10: 0.97}
```

## GPU support (experimental)

The repository contains experimental CUDA and ROCm kernels for batched L2 distance,
available from Rust behind the `cuda` / `rocm` Cargo features. They are not used by the
Python classes, are not included in the published wheels, and are not covered by
continuous integration.
