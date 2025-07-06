

## 📋 Changelog Structure


Under each version, group entries into categories:

* **Added**      for new features
* **Changed**    for updates to existing behavior
* **Deprecated** for soon-to-be-removed features
* **Removed**    for now-removed features
* **Fixed**      for bug fixes
* **Security**   for vulnerability patches

---




# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org).

## [Unreleased]

### Added
- Support for Manhattan (L1) distance in `Distance.MANHATTAN`.
- New `remove(ids: List[int])` method to delete entries by ID.
- GIL-release in `search()` and `search_batch()` for lower Python-side latency.

### Changed
- Bumped `rust_annie` version to **0.1.1**.

## [0.1.1] – 2025-05-20

### Added
- Manhattan (L1) distance support:
  ```python
  from rust_annie import Distance
  idx = AnnIndex(16, Distance.MANHATTAN)


* `Distance.MANHATTAN` class attribute and `__repr__` value.

### Fixed

* Correctly annotate `.collect::<Vec<_>>()` in batch search.
* Removed `into_pyerr()` misuse in `search()`.

## \[0.1.0] – 2025-05-16

### Added

* Initial release with Euclidean (L2) and Cosine distances.
* `AnnIndex`, `search()`, `search_batch()`, `add()`, `save()`, `load()` APIs.
* SIMD‐free brute-force search accelerated by **Rayon**.
* Thread-safe wrapper `ThreadSafeAnnIndex` with GIL release.

### Changed

* Logging improvements in CI workflow.
* Performance optimizations: cached norms, GIL release, parallel loops.

### Fixed

* Various build errors on Windows and macOS.

## \[0.0.1] – 2025-05-10

### Added

* Prototype implementation of brute-force k-NN index in Rust.
 this in place, anyone browsing your repo or reading release notes on PyPI will immediately see what changed in each version.

