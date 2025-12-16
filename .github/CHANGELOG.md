# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org).

## [Unreleased]

## [0.2.5] – 2025-12-11

### Added
- Support for Manhattan (L1) distance in `Distance.MANHATTAN`
- New `remove(ids: List[int])` method to delete entries by ID
- GIL-release in `search()` and `search_batch()` for lower Python-side latency
- Metadata-aware filtering and typed predicate pushdown ([#209](https://github.com/arnavk23/Annie/pull/209))
- RAII GPU memory management to fix critical memory leaks ([#191](https://github.com/arnavk23/Annie/pull/191))
- Batch addition methods in AnnIndex and HnswIndex ([#145](https://github.com/arnavk23/Annie/pull/145))
- Filtering capabilities ([#147](https://github.com/arnavk23/Annie/pull/147))
- Plugin distance metrics ([#107](https://github.com/arnavk23/Annie/pull/107))
- KNN monitoring capabilities ([#109](https://github.com/arnavk23/Annie/pull/109))
- Verification functionality ([#99](https://github.com/arnavk23/Annie/pull/99))
- GPU backend support with CUDA and ROCm ([#135](https://github.com/arnavk23/Annie/pull/135), [#85](https://github.com/arnavk23/Annie/pull/85))
- Benchmark suite with criterion ([#143](https://github.com/arnavk23/Annie/pull/143))
- Path validation in save and load methods ([#127](https://github.com/arnavk23/Annie/pull/127))
- Enhanced error handling and validation ([#134](https://github.com/arnavk23/Annie/pull/134), [#137](https://github.com/arnavk23/Annie/pull/137))
- Serialization support for HnswIndex ([#132](https://github.com/arnavk23/Annie/pull/132))
- Docstrings with text_signature to AnnIndex methods ([#101](https://github.com/arnavk23/Annie/pull/101))

### Changed
- Bumped `rust_annie` version to **0.2.5**
- Upgraded PyO3 to 0.25 with API compatibility fixes ([#221](https://github.com/arnavk23/Annie/pull/221))
- Upgraded ndarray to 0.16.0 ([#112](https://github.com/arnavk23/Annie/pull/112))
- Updated syn to 2.0+ versions
- Updated serde to 1.0.228
- Refactored shared distance helper utilities ([#82](https://github.com/arnavk23/Annie/pull/82))
- Optimized HNSW configuration ([#137](https://github.com/arnavk23/Annie/pull/137))
- Enhanced CPU backend selection ([#95](https://github.com/arnavk23/Annie/pull/95))
- Macros implementation for improved type handling ([#93](https://github.com/arnavk23/Annie/pull/93))
- Python binding enhancements for `len()` and `dim` properties ([#80](https://github.com/arnavk23/Annie/pull/80))
- Updated CI configuration with Python 3.12 support ([#190](https://github.com/arnavk23/Annie/pull/190))
- Refactored auto-release workflow for Python package ([#234](https://github.com/arnavk23/Annie/pull/234))

### Fixed
- Fixed path traversal vulnerability in file operations ([#204](https://github.com/arnavk23/Annie/pull/204))
- Memory safety violations in GPU backend with bytemuck ([#210](https://github.com/arnavk23/Annie/pull/210))
- Compilation errors preventing CI checks ([#212](https://github.com/arnavk23/Annie/pull/212))
- NaN handling in index operations ([#51](https://github.com/arnavk23/Annie/pull/51))
- PyErr error standardization ([#23](https://github.com/arnavk23/Annie/pull/23))
- Benchmark output JSON formatting ([#28](https://github.com/arnavk23/Annie/pull/28))
- HNSW integration issues ([#25](https://github.com/arnavk23/Annie/pull/25))
- Poisoning issue in thread-safe index ([#192](https://github.com/arnavk23/Annie/pull/192))
- Integer filter implementation issues ([#181](https://github.com/arnavk23/Annie/pull/181))
- Unused imports cleanup ([#216](https://github.com/arnavk23/Annie/pull/216))

### Security
- Fix critical path traversal vulnerability in file operations ([#204](https://github.com/arnavk23/Annie/pull/204))
- Fix memory safety violations in GPU backend ([#210](https://github.com/arnavk23/Annie/pull/210))
- Deployment security enhancements ([#177](https://github.com/arnavk23/Annie/pull/177))

### Documentation
- Update examples and index for fuzz testing integration ([#100](https://github.com/arnavk23/Annie/pull/100))
- Update AnnIndex documentation and examples ([#102](https://github.com/arnavk23/Annie/pull/102))
- Update AnnIndex and HnswIndex documentation for API changes ([#104](https://github.com/arnavk23/Annie/pull/104))
- Update examples.md with benchmarking instructions ([#106](https://github.com/arnavk23/Annie/pull/106))
- Update API documentation for custom distance metrics ([#108](https://github.com/arnavk23/Annie/pull/108))
- Update concurrency and API documentation ([#113](https://github.com/arnavk23/Annie/pull/113))
- Update documentation for path validation ([#128](https://github.com/arnavk23/Annie/pull/128))
- Update HnswIndex documentation for serialization ([#133](https://github.com/arnavk23/Annie/pull/133))
- Update documentation for GPU support ([#136](https://github.com/arnavk23/Annie/pull/136))
- Update for new distance metrics ([#140](https://github.com/arnavk23/Annie/pull/140))
- Update HNSW Index documentation for PyHnswConfig ([#138](https://github.com/arnavk23/Annie/pull/138))
- Update HnswIndex, ThreadSafeAnnIndex and Concurrency documentation ([#142](https://github.com/arnavk23/Annie/pull/142))
- Update Benchmark workflow and monitoring documentation ([#144](https://github.com/arnavk23/Annie/pull/144))
- Update filtering and AnnIndex documentation ([#148](https://github.com/arnavk23/Annie/pull/148))
- Update API documentation for AnnIndex, HnswIndex, ThreadSafeAnnIndex ([#156](https://github.com/arnavk23/Annie/pull/156))
- Update examples and filtering documentation ([#158](https://github.com/arnavk23/Annie/pull/158))
- Update GPU backend creation and error handling documentation ([#160](https://github.com/arnavk23/Annie/pull/160))
- Update HnswIndex and AnnIndex documentation for new methods ([#162](https://github.com/arnavk23/Annie/pull/162))
- Update main documentation index for rust_annie version 0.2.3 ([#174](https://github.com/arnavk23/Annie/pull/174))
- Update API and concurrency documentation ([#180](https://github.com/arnavk23/Annie/pull/180))
- Update GPU Performance and API documentation ([#182](https://github.com/arnavk23/Annie/pull/182))
- Update hnsw_index documentation ([#187](https://github.com/arnavk23/Annie/pull/187))
- Update security audit documentation ([#189](https://github.com/arnavk23/Annie/pull/189))
- Documentation updates for error handling ([#193](https://github.com/arnavk23/Annie/pull/193))
- Update documentation for metadata-aware filtering ([#198](https://github.com/arnavk23/Annie/pull/198), [#200](https://github.com/arnavk23/Annie/pull/200))
- Update hnsw_index.md for serde compatibility ([#203](https://github.com/arnavk23/Annie/pull/203))
- Update GPU Performance documentation for Bytemuck ([#214](https://github.com/arnavk23/Annie/pull/214))
- Update security audit for new path validation checks ([#217](https://github.com/arnavk23/Annie/pull/217))
- Update documentation for python3-dll-a dependency ([#220](https://github.com/arnavk23/Annie/pull/220))

### Dependencies
- Bump proc-macro2 from 1.0.95 to 1.0.103 ([#173](https://github.com/arnavk23/Annie/pull/173), [#223](https://github.com/arnavk23/Annie/pull/223))
- Bump rayon from 1.10.0 to 1.11.0 ([#185](https://github.com/arnavk23/Annie/pull/185), [#9](https://github.com/arnavk23/Annie/pull/9))
- Bump syn from 2.0.104 to 2.0.111+ ([#184](https://github.com/arnavk23/Annie/pull/184), [#222](https://github.com/arnavk23/Annie/pull/222), [#225](https://github.com/arnavk23/Annie/pull/225), [#226](https://github.com/arnavk23/Annie/pull/226))
- Bump thiserror from 1.0.69 to 2.0.17 ([#183](https://github.com/arnavk23/Annie/pull/183), [#207](https://github.com/arnavk23/Annie/pull/207))
- Bump rand from 0.8.5 to 0.9.2 ([#126](https://github.com/arnavk23/Annie/pull/126), [#153](https://github.com/arnavk23/Annie/pull/153), [#163](https://github.com/arnavk23/Annie/pull/163))
- Bump criterion from 0.5.1 to 0.8.1 ([#125](https://github.com/arnavk23/Annie/pull/125), [#154](https://github.com/arnavk23/Annie/pull/154), [#164](https://github.com/arnavk23/Annie/pull/164), [#228](https://github.com/arnavk23/Annie/pull/228), [#230](https://github.com/arnavk23/Annie/pull/230))
- Bump bit-vec from 0.6.3 to 0.8.0 ([#152](https://github.com/arnavk23/Annie/pull/152))
- Bump quote from 1.0.40 to 1.0.42 ([#205](https://github.com/arnavk23/Annie/pull/205), [#224](https://github.com/arnavk23/Annie/pull/224))
- Bump serde from 1.0.219 to 1.0.228 ([#201](https://github.com/arnavk23/Annie/pull/201), [#202](https://github.com/arnavk23/Annie/pull/202), [#206](https://github.com/arnavk23/Annie/pull/206))
- Bump bytemuck from 1.23.2 to 1.24.0 ([#213](https://github.com/arnavk23/Annie/pull/213))
- Bump half from 2.6.0 to 2.7.1 ([#215](https://github.com/arnavk23/Annie/pull/215), [#219](https://github.com/arnavk23/Annie/pull/219))
- Bump hnsw_rs from 0.3.2 to 0.3.3 ([#227](https://github.com/arnavk23/Annie/pull/227))
- Bump hip-rs to 1.0.0 ([#87](https://github.com/arnavk23/Annie/pull/87))

## [0.2.3] – 2025-11-01

### Added
- HNSW backend integration ([#34](https://github.com/arnavk23/Annie/pull/34))
- Distance tests ([#20](https://github.com/arnavk23/Annie/pull/20))
- ann_bench.rs benchmark ([#105](https://github.com/arnavk23/Annie/pull/105))

### Changed
- Enhanced HNSW integration demo ([#25](https://github.com/arnavk23/Annie/pull/25))

## [0.2.1] – 2025-10-15

### Changed
- Version bump to 0.2.1 ([#130](https://github.com/arnavk23/Annie/pull/130))

## [0.1.1] – 2025-05-20

### Added
- Manhattan (L1) distance support
- `Distance.MANHATTAN` class attribute and `__repr__` value

### Fixed
- Correctly annotate `.collect::<Vec<_>>()` in batch search
- Removed `into_pyerr()` misuse in `search()`

## [0.1.0] – 2025-05-16

### Added
- Initial release with Euclidean (L2) and Cosine distances
- `AnnIndex`, `search()`, `search_batch()`, `add()`, `save()`, `load()` APIs
- SIMD-free brute-force search accelerated by **Rayon**
- Thread-safe wrapper `ThreadSafeAnnIndex` with GIL release

### Changed
- Logging improvements in CI workflow
- Performance optimizations: cached norms, GIL release, parallel loops

### Fixed
- Various build errors on Windows and macOS

## [0.0.1] – 2025-05-10

### Added
- Prototype implementation of brute-force k-NN index in Rust
