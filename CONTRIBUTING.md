# Contributing to rust-annie

Thanks for your interest in improving rust-annie! This document explains how to report
problems, ask for help, and contribute code.

## Reporting bugs and requesting features

Open an issue at <https://github.com/Programmers-Paradise/Annie/issues> using the bug report or feature
request template. For bugs, please include:

- the rust-annie version (`python -c "import rust_annie; print(rust_annie.__version__)"`),
  Python version and operating system;
- a minimal, self-contained script that reproduces the problem;
- what you expected to happen and what happened instead (full traceback if any).

Security issues should be reported privately via GitHub's
["Report a vulnerability"](https://github.com/Programmers-Paradise/Annie/security/advisories/new) form
rather than a public issue.

## Getting help

Questions about using the library are welcome as GitHub issues (choose "Question") or in
[Discussions](https://github.com/Programmers-Paradise/Annie/discussions) if enabled. Please check the
[user guide](docs/user-guide.md) and [API reference](docs/api.md) first.

## Development setup

You need Python ≥ 3.9 and a Rust toolchain ≥ 1.74 (install via <https://rustup.rs>).

```bash
git clone https://github.com/Programmers-Paradise/Annie.git
cd Annie
python -m venv .venv && source .venv/bin/activate
pip install maturin pytest hypothesis ruff
maturin develop --release        # build and install into the virtualenv
```

Re-run `maturin develop --release` after changing Rust code.

## Running the checks

CI runs all of the following; please run them before opening a pull request.

```bash
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
cargo test                      # Rust unit and integration tests
pytest tests/python             # Python tests, including every code block in the docs
ruff check python tests benchmarks examples
ruff format --check python tests benchmarks examples
for f in examples/*.py; do python "$f"; done
```

Optional:

```bash
cargo bench                                          # Criterion micro-benchmarks
python benchmarks/compare.py --dataset random-xs     # comparison with hnswlib/FAISS
cargo +nightly fuzz run input_validation             # fuzzing (needs cargo-fuzz)
```

## Project layout

| Path | Contents |
|---|---|
| `src/index.rs` | exact (brute-force) index |
| `src/hnsw_index.rs` | HNSW index (graph from the `hnsw_rs` crate) |
| `src/concurrency.rs` | thread-safe wrapper |
| `src/search.rs` | shared distance, top-k and validation helpers |
| `src/metrics.rs`, `src/distance_registry.rs` | built-in and custom metrics |
| `src/filters.rs`, `src/storage.rs`, `src/monitoring.rs` | filters, file format, metrics |
| `src/gpu/` | experimental CUDA/ROCm kernels (not used by the Python API) |
| `python/rust_annie/` | Python package and type stubs |
| `tests/python/`, `tests/*.rs` | Python and Rust tests |
| `benchmarks/`, `benches/` | comparison script, Criterion benchmarks |

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Add tests for new behaviour and update the docs (`README.md`, `docs/`) and the type
   stubs (`python/rust_annie/_rust_annie.pyi`) if the Python API changes. Code blocks in
   the docs are executed by the test suite.
3. Add an entry under "Unreleased" in `CHANGELOG.md`.
4. Make sure all checks above pass, then open a pull request describing the change and
   its motivation.

A maintainer will review your pull request; please be patient and responsive to
feedback. By contributing you agree that your contributions are licensed under the
project's MIT license.

## Releasing (maintainers)

1. Update `version` in `Cargo.toml` and move the "Unreleased" changelog entries under a
   new `## [X.Y.Z] - YYYY-MM-DD` heading.
2. Merge to `main`, then tag: `git tag vX.Y.Z && git push origin vX.Y.Z`.
3. The `Release` workflow checks the tag against `Cargo.toml` and the changelog, builds
   and tests wheels for all platforms, publishes to PyPI via trusted publishing and creates
   the GitHub release.

## Code of conduct

Everyone participating in this project is expected to follow the
[Code of Conduct](CODE_OF_CONDUCT.md).
