# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/) (before 1.0, minor versions may break the API).

## [Unreleased]

## [0.3.0] - 2026-11-06

This release fixes many correctness problems and makes the documented API match the
implementation. It contains breaking changes; see "Migrating from 0.2" below.

### Fixed
- `PyHnswIndex`, `ThreadSafeAnnIndex`, `Index`, `HnswConfig`, `register_metric` and
  `list_metrics` were not importable from `rust_annie` (`__all__` only listed
  `AnnIndex` and `Distance`).
- `AnnIndex.search` and `search_batch` required a `filter` argument; it is now optional.
- `AnnIndex.search_batch` always used Euclidean distance regardless of the index metric,
  and failed when `k` exceeded the number of results.
- `AnnIndex.search` returned squared Euclidean distances (and small non-zero values for
  identical vectors); all methods now return true distances, consistently.
- `Distance.ANGULAR` returned cosine distance instead of the angle.
- Indexes using custom metrics silently returned no results; Python exceptions raised in
  custom metrics were swallowed.
- `AnnIndex.save` and `load` used incompatible formats, so saved indexes could never be
  loaded; `save` also appended `.bin` to the given path.
- `PyHnswIndex.search` returned internal positions instead of the user's IDs, returned no
  distances, and ignored `ef`; `PyHnswIndex.load` was not implemented.
- `PyHnswIndex` passed construction parameters to the graph in the wrong order, silently
  dropped vectors beyond `max_elements`, and an `m` above 256 terminated the Python
  process.
- `Index("hnsw", ...)` returned all-zero distances.
- Saving to absolute paths or outside the working directory was rejected.
- `Filter.and`/`or`/`not` could not be called from Python (reserved words), and
  `Filter.from_py_callable` rejected every item.
- `Distance` values did not compare equal (`Distance.COSINE == Distance.COSINE` was
  `False`) and were unhashable.
- Duplicate IDs within one `add` call were not detected; NaN/infinite values were accepted.
- `update_recall_estimate` and `enable_metrics` recorded nothing.
- `PyHnswIndex` searches with selective filters could miss matching vectors (graph
  traversal stopped before reaching them); such searches now fall back to an exact scan
  of the matching vectors.

### Added
- `AnnIndex.remove`, `update` and `compact` (previously raised `NotImplementedError`),
  with automatic compaction.
- `PyHnswIndex`: cosine and Manhattan metrics, `remove`, filtered search, `search_batch`,
  per-query `ef`, `ef_search` property, persistence, and constructor keyword arguments.
- Filters on every index type, including arbitrary Python callables (`filter=lambda i: ...`).
- `Index`: `add`, `search_batch`, `remove`, `save` and `load` (detects the index kind).
- Inputs may be any array-like (float64 arrays, lists, int32 IDs) and are converted.
- `ids()`, `id in index`, `metric` property; `__version__`.
- Versioned file format with a header identifying the index kind.
- Type stubs (`py.typed`), user guide, API reference and runnable examples.
- Tests against NumPy reference results (including property-based tests with
  Hypothesis), a fuzz target for the index API, and execution of every documentation
  code block in CI.
- `benchmarks/compare.py`: recall/throughput comparison with hnswlib and FAISS on
  ann-benchmarks datasets.

### Changed
- Errors now raise specific exceptions (`ValueError`, `KeyError`, `OSError`,
  `TypeError`, `RuntimeError`) instead of bare `Exception`.
- Searching an empty index returns empty arrays instead of raising.
- Batch results are always shaped `(n, k)`, padded with `-1` / `inf`.
- The GIL is released during searches with built-in metrics; `ThreadSafeAnnIndex`
  acquires its lock without holding the GIL (prevents deadlocks).
- HNSW defaults: `m=16`, `ef_construction=200`, `ef_search=50`.
- Maximum dimensionality raised from 4 096 to 65 536; the 1 000 000-vector limit was removed.
- Minimum Python version is 3.9; building from source needs Rust 1.85 or newer.
- Exact search is several times faster: vectors are stored contiguously, distance
  kernels are vectorised, the top-k buffer is bounded, and small searches avoid
  thread-pool overhead.

### Removed
- Metadata schema methods (`py_set_metadata_schema`, `py_add_with_metadata`,
  `py_search_filtered`) and the `MetadataType`/`MetadataField`/`MetadataValue` classes:
  predicates were never evaluated. Use filters instead.
- `update_boolean_filter` / `get_boolean_filter` (stored filters were never used) and
  `Filter.from_py_callable` (pass the callable directly).
- `PyHnswIndex.add_item` and `build`.
- Unused internal modules, the per-commit benchmark dashboard, and committed benchmark
  results.

### Migrating from 0.2
- Index files written by 0.2 cannot be loaded (0.2 could not load them either); rebuild
  and save again.
- `save(path)` writes to exactly `path`; previously `path + ".bin"`.
- Replace `Filter.and(...)`, `Filter.or(...)`, `Filter.not(...)` with `and_`, `or_`, `not_`.
- `PyHnswIndex.search` now returns `(ids, distances)` instead of a list of positions.
- Code catching `Exception` keeps working; code matching on the message
  prefix `RustAnnError [...]` also keeps working.

## [0.2.4] - 2025-08-29
## [0.2.3] - 2025-08-05
## [0.2.1] - 2025-07-14
## [0.2.0] - 2025-06-21
## [0.1.3] - 2025-06-15
## [0.1.2] - 2025-05-17
## [0.1.1] - 2025-05-16

Earlier releases are described on the
[GitHub releases page](https://github.com/Programmers-Paradise/Annie/releases).

[Unreleased]: https://github.com/Programmers-Paradise/Annie/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Programmers-Paradise/Annie/compare/v0.2.4...v0.3.0
