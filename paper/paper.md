---
title: 'rust-annie: exact and approximate nearest-neighbour search with arbitrary metrics for Python'
tags:
  - Python
  - Rust
  - nearest neighbour search
  - similarity search
  - HNSW
  - metric learning
authors:
  - name: Arnav Kapoor
    orcid: 0009-0007-9818-7908
    corresponding: true
    affiliation: 1
  - name: Abhinav Shukla
    affiliation: 2
affiliations:
  - name: Department of Electrical Engineering and Computer Science, Indian Institute of Science Education and Research Bhopal, India
    index: 1
  - name: Chhattisgarh Swami Vivekanand Technical University, Bhilai, India
    index: 2
date: 6 November 2026
bibliography: paper.bib
---

# Summary

Finding the *k* stored vectors closest to a query vector is a basic operation in
scientific computing: it underlies nearest-neighbour classification and regression,
clustering, outlier detection, duplicate detection, and retrieval over learned
embeddings. `rust-annie` is a Python library, implemented in Rust, that provides exact
*k*-nearest-neighbour search under a wide range of distance functions — including
distance functions written by the user in Python — and approximate search with
hierarchical navigable small-world (HNSW) graphs [@malkov2018hnsw], behind a single
NumPy-based [@harris2020numpy] interface. Indexes are keyed by user-supplied integer
identifiers and support insertion, removal, in-place update, filtered search,
persistence and safe concurrent use from Python threads.

# Statement of need

Researchers often need nearest neighbours under a distance that is specific to their
problem: a Mahalanobis distance [@mahalanobis1936] estimated from their data, a
weighted or domain-specific dissimilarity, or a set-based measure on binary
fingerprints. The established high-performance libraries — FAISS
[@johnson2019faiss; @douze2024faiss], hnswlib [@hnswlib], Annoy [@annoy], ScaNN
[@guo2020scann] — are optimised for a small set of metrics (typically Euclidean and inner
product) and cannot evaluate an arbitrary Python function. General-purpose tools that do
accept arbitrary metrics, such as `scipy.spatial.distance.cdist` [@virtanen2020scipy] or
scikit-learn's `NearestNeighbors` [@pedregosa2011sklearn], are not designed as mutable
indexes: they offer no stable identifiers, removal, filtering or persistence, and
switching to an approximate method later means switching libraries.

`rust-annie` targets this gap. A user can prototype with a Python-defined metric on an
exact index, move to a native metric (nine are built in, computed in parallel without
holding Python's global interpreter lock), and, where approximate results are acceptable,
switch to an HNSW index with the same identifiers, filters, persistence and
result conventions. The intended audience is researchers and research software
engineers working with vector data of up to a few million items on a single machine who
want this flexibility without assembling several libraries.

# State of the field

FAISS provides exact and approximate search, including quantisation-based indexes and
GPU support, for Euclidean and inner-product similarity (with a few additional metrics
for brute-force search). hnswlib offers a fast HNSW implementation for L2, inner product
and cosine. Annoy builds random-projection forests for static collections. USearch
[@usearch] provides an HNSW engine with several built-in metrics and support for
user-defined metrics compiled from C or Numba. scikit-learn and SciPy accept arbitrary
Python callables as metrics but compute neighbours from scratch for each call or build
static tree structures. `rust-annie` does not aim to compete with FAISS or hnswlib on raw
approximate-search throughput — its HNSW implementation builds on the `hnsw_rs` crate
[@hnswrs] — but combines exact search with arbitrary (including pure-Python) metrics,
approximate search, mutability, and filtering in one interface.

# Software design

The core is written in Rust and exposed to Python with PyO3 [@pyo3]. `AnnIndex` stores
vectors contiguously with cached norms and performs exact search by scanning all stored
vectors, in parallel with Rayon for large indexes, keeping the best *k* in a bounded
buffer; ties are broken by identifier so results are deterministic. Batch queries are parallelised across queries.
For built-in metrics the Python global interpreter lock is released during search;
for Python-defined metrics the scan runs on the calling thread and each distance is a
Python call, with exceptions propagated to the caller. `PyHnswIndex` wraps `hnsw_rs`
graphs (one per supported metric), maps graph-internal identifiers to user identifiers,
implements removal as a tombstone filter applied during graph traversal, switches to an
exact scan of the matching vectors when a filter is too selective for graph traversal
to find them reliably, and rebuilds
the graph on load from a versioned file format. Filters (identifier ranges, sets,
bitmasks, their boolean combinations, or Python predicates) are applied during the scan
or graph traversal. `ThreadSafeAnnIndex` wraps the exact index in a readers–writer lock
that is only ever acquired with the interpreter lock released, which rules out
lock-ordering deadlocks between Python threads.

The test suite compares every metric and code path against NumPy reference
implementations, including property-based tests, executes every code example in the
documentation, and fuzzes the index API.

# Performance

Exact search scans vectors stored contiguously in memory with distance kernels that
the compiler vectorises, keeps a bounded top-*k* buffer, and splits large scans across
cores with Rayon, without holding the Python global interpreter lock.
\autoref{fig:sift} compares `rust-annie` with hnswlib, FAISS and NumPy on the SIFT-1M
benchmark [@aumuller2020annbenchmarks] on a four-core laptop. Answering one query at a
time, exact search in `rust-annie` is about eight times faster than NumPy and faster than
FAISS's flat index restricted to one thread. For large batches of queries FAISS is about
ten times faster, because it computes distances as a BLAS matrix product. `rust-annie`'s
HNSW index reaches the same recall as hnswlib and FAISS for the same parameters, but it
answers queries about five times more slowly and builds about three times more slowly,
so users who need maximum approximate-search throughput for Euclidean or inner-product
similarity should prefer those libraries. `benchmarks/compare.py` in the repository
reproduces these measurements.

![Recall@10 against queries per second (one query at a time, log scale) on SIFT-1M
(10⁶ vectors, 128 dimensions, 1 000 queries; HNSW with $M = 16$,
$ef_\text{construction} = 200$, $ef$ from 10 to 320). Intel Core i5-1135G7, 4 cores.
\label{fig:sift}](figures/sift-128-euclidean.png)

# Research impact statement

`rust-annie` is developed in the open by Programmers' Paradise, the technical society
of Chhattisgarh Swami Vivekanand Technical University, and has received contributions
from 14 people since its first release in May 2025. It is distributed on PyPI with
pre-built wheels for Linux, macOS and Windows. At the time of writing we are not aware
of published research that uses it. This release makes the library suitable for such
use: it corrects defects in earlier versions, documents the behaviour of every metric
and method, and validates results against NumPy reference implementations in
continuous integration. We will document research use as it is reported, and we invite
users to cite the software using the metadata in the repository.

# AI usage disclosure

Generative AI tools contributed to this project. GitHub Copilot's coding agent authored
29 commits (August–December 2025) to continuous-integration and release configuration,
and an automated pull-request review bot (Entelligence) committed edits to earlier
versions of the documentation, which has since been rewritten. Claude Code (Anthropic)
was used to prepare version 0.3.0: correcting the defects listed in the changelog,
writing the test suite, benchmark script, documentation and an initial draft of this
paper. All AI-assisted changes were reviewed, tested and edited by the authors, who take
full responsibility for the software and this paper.

# Acknowledgements

We thank Gaurav Karakoti, Akash Bisht and the other contributors to `rust-annie`;
Aksh Agrawal and Vishal Maurya for issue reports and code review; and the authors of
`hnsw_rs`, PyO3 and NumPy, on which this work builds.

# References
