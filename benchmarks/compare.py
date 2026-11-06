#!/usr/bin/env python3
"""Recall / throughput comparison of rust_annie against other k-NN libraries.

Follows the ann-benchmarks methodology (Aumüller et al., 2020): standard
HDF5 datasets with precomputed ground truth, recall@k, and queries issued one
at a time on a single thread. A batch mode measures multi-threaded throughput.

Examples
--------
Quick synthetic run (no download):

    python benchmarks/compare.py --dataset random-xs

Standard dataset (downloaded from ann-benchmarks.com and cached):

    python benchmarks/compare.py --dataset sift-128-euclidean --plot

Optional dependencies: ``pip install hnswlib faiss-cpu h5py matplotlib``.
Libraries that are not installed are skipped.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

DATASETS = {
    "sift-128-euclidean": "euclidean",
    "glove-100-angular": "cosine",
    "fashion-mnist-784-euclidean": "euclidean",
    "nytimes-256-angular": "cosine",
}
SYNTHETIC = {"random-xs": (10_000, 32, 200), "random-s": (100_000, 64, 500)}
URL = "https://ann-benchmarks.com/{}.hdf5"


@dataclass
class Result:
    dataset: str
    library: str
    algorithm: str
    params: str
    k: int
    recall: float
    qps_single: float
    qps_batch: float
    build_s: float


def load_dataset(name: str, cache: Path, k: int):
    if name in SYNTHETIC:
        n, d, nq = SYNTHETIC[name]
        rng = np.random.default_rng(0)
        train = rng.standard_normal((n, d)).astype(np.float32)
        test = rng.standard_normal((nq, d)).astype(np.float32)
        # exact ground truth computed blockwise with NumPy
        sq = (train**2).sum(1)
        gt = np.empty((nq, k), dtype=np.int64)
        for s in range(0, nq, 100):
            q = test[s : s + 100]
            d2 = sq[None, :] - 2 * q @ train.T
            gt[s : s + 100] = np.argsort(d2, axis=1)[:, :k]
        return train, test, gt, "euclidean"

    import h5py

    path = cache / f"{name}.hdf5"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URL.format(name)} -> {path}", file=sys.stderr)
        urllib.request.urlretrieve(URL.format(name), path)
    with h5py.File(path, "r") as f:
        train = np.asarray(f["train"], dtype=np.float32)
        test = np.asarray(f["test"], dtype=np.float32)
        gt = np.asarray(f["neighbors"])[:, :k].astype(np.int64)
    return train, test, gt, DATASETS[name]


def recall(found: np.ndarray, gt: np.ndarray) -> float:
    hits = sum(len(set(f[f >= 0].tolist()) & set(g.tolist())) for f, g in zip(found, gt))
    return hits / gt.size


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


def run_queries(search_one, search_batch, test, k):
    found = np.empty((len(test), k), dtype=np.int64)

    def single():
        for i, q in enumerate(test):
            ids = search_one(q, k)
            found[i, : len(ids)] = ids
            found[i, len(ids) :] = -1

    _, t_single = timed(single)
    _, t_batch = timed(lambda: search_batch(test, k))
    return found, len(test) / t_single, len(test) / t_batch


# ---------------------------------------------------------------------------
# Library adapters. Each yields (algorithm, params, build_s, search_one, search_batch)
# ---------------------------------------------------------------------------


def rust_annie_runs(train, metric, args):
    import rust_annie as ra

    dist = ra.Distance.EUCLIDEAN if metric == "euclidean" else ra.Distance.COSINE
    ids = np.arange(len(train), dtype=np.int64)

    if len(train) <= args.max_exact:
        idx = ra.AnnIndex(train.shape[1], dist)
        _, build = timed(lambda: idx.add(train, ids))
        yield (
            "brute-force",
            "",
            build,
            lambda q, k: idx.search(q, k)[0],
            lambda Q, k: idx.search_batch(Q, k),
        )

    h = ra.PyHnswIndex(train.shape[1], metric=dist, m=args.m, ef_construction=args.ef_construction)
    _, build = timed(lambda: h.add(train, ids))
    for ef in args.ef:
        yield (
            "hnsw",
            f"M={args.m},efC={args.ef_construction},ef={ef}",
            build,
            lambda q, k, ef=ef: h.search(q, k, ef=ef)[0],
            lambda Q, k, ef=ef: h.search_batch(Q, k, ef=ef),
        )


def hnswlib_runs(train, metric, args):
    import hnswlib

    space = "l2" if metric == "euclidean" else "cosine"
    p = hnswlib.Index(space=space, dim=train.shape[1])

    def build():
        p.init_index(max_elements=len(train), M=args.m, ef_construction=args.ef_construction, random_seed=0)
        p.add_items(train, np.arange(len(train)), num_threads=-1)

    _, t = timed(build)
    for ef in args.ef:

        def one(q, k, ef=ef):
            p.set_ef(max(ef, k))
            p.set_num_threads(1)
            return p.knn_query(q, k)[0][0]

        def batch(Q, k, ef=ef):
            p.set_ef(max(ef, k))
            p.set_num_threads(os.cpu_count() or 1)
            return p.knn_query(Q, k)

        yield ("hnsw", f"M={args.m},efC={args.ef_construction},ef={ef}", t, one, batch)


def faiss_runs(train, metric, args):
    import faiss

    data = train
    if metric == "cosine":
        data = train / np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-12)

    def prep(q):
        if metric == "cosine":
            q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
        return np.ascontiguousarray(np.atleast_2d(q), dtype=np.float32)

    def make(index):
        def one(q, k):
            faiss.omp_set_num_threads(1)
            return index.search(prep(q), k)[1][0]

        def batch(Q, k):
            faiss.omp_set_num_threads(os.cpu_count() or 1)
            return index.search(prep(Q), k)

        return one, batch

    faiss.omp_set_num_threads(os.cpu_count() or 1)
    if len(train) <= args.max_exact:
        flat = faiss.IndexFlatL2(train.shape[1])
        _, t = timed(lambda: flat.add(data))
        yield ("brute-force", "", t, *make(flat))

    h = faiss.IndexHNSWFlat(train.shape[1], args.m)
    h.hnsw.efConstruction = args.ef_construction
    _, t = timed(lambda: h.add(data))
    for ef in args.ef:
        one, batch = make(h)

        def one_ef(q, k, ef=ef, one=one):
            h.hnsw.efSearch = max(ef, k)
            return one(q, k)

        def batch_ef(Q, k, ef=ef, batch=batch):
            h.hnsw.efSearch = max(ef, k)
            return batch(Q, k)

        yield ("hnsw", f"M={args.m},efC={args.ef_construction},ef={ef}", t, one_ef, batch_ef)


def numpy_runs(train, metric, args):
    if len(train) > args.max_exact:
        return
    data = train
    if metric == "cosine":
        data = train / np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-12)
    sq = (data**2).sum(1)

    def one(q, k):
        if metric == "cosine":
            q = q / max(np.linalg.norm(q), 1e-12)
        d = sq - 2 * data @ q
        part = np.argpartition(d, k)[:k]
        return part[np.argsort(d[part])]

    def batch(Q, k):
        return np.stack([one(q, k) for q in Q])

    yield ("brute-force", "", 0.0, one, batch)


LIBRARIES = {
    "rust_annie": rust_annie_runs,
    "hnswlib": hnswlib_runs,
    "faiss": faiss_runs,
    "numpy": numpy_runs,
}


def plot(results: list[Result], path: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    groups: dict[str, list[Result]] = {}
    for r in results:
        groups.setdefault(f"{r.library} ({r.algorithm})", []).append(r)
    for label, rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: r.recall)
        ax.plot([r.recall for r in rs], [r.qps_single for r in rs], "o-", label=label)
    ax.set_xlabel(f"Recall@{results[0].k}")
    ax.set_ylabel("Queries per second (1 thread, log scale)")
    ax.set_yscale("log")
    ax.set_title(results[0].dataset)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"wrote {path}", file=sys.stderr)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="random-xs", choices=[*SYNTHETIC, *DATASETS])
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--m", type=int, default=16)
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--ef", type=int, nargs="+", default=[10, 20, 40, 80, 160, 320])
    p.add_argument("--max-queries", type=int, default=1000)
    p.add_argument(
        "--max-exact", type=int, default=2_000_000, help="skip brute force above this many vectors"
    )
    p.add_argument("--libraries", nargs="+", default=list(LIBRARIES), choices=list(LIBRARIES))
    p.add_argument("--cache", type=Path, default=Path("~/.cache/rust_annie_bench").expanduser())
    p.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    p.add_argument("--plot", action="store_true")
    args = p.parse_args(argv)

    train, test, gt, metric = load_dataset(args.dataset, args.cache, args.k)
    test, gt = test[: args.max_queries], gt[: args.max_queries]
    print(
        f"{args.dataset}: {train.shape[0]} x {train.shape[1]} ({metric}), {len(test)} queries",
        file=sys.stderr,
    )

    results = []
    for lib in args.libraries:
        try:
            runs = list(LIBRARIES[lib](train, metric, args))
        except ImportError as e:
            print(f"skipping {lib}: {e}", file=sys.stderr)
            continue
        for algorithm, params, build, one, batch in runs:
            found, qps1, qpsb = run_queries(one, batch, test, args.k)
            r = Result(args.dataset, lib, algorithm, params, args.k, recall(found, gt), qps1, qpsb, build)
            results.append(r)
            print(
                f"{lib:10s} {algorithm:11s} {params:28s} recall={r.recall:.4f} "
                f"qps(1t)={qps1:9.0f} qps(batch)={qpsb:9.0f} build={build:.2f}s",
                file=sys.stderr,
            )

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / f"{args.dataset}.csv"
    with open(csv_path, "w", newline="") as f:
        cpu = platform.processor() or platform.machine()
        f.write(f"# {platform.platform()} | {cpu} | {os.cpu_count()} CPUs\n")
        w = csv.DictWriter(f, fieldnames=list(asdict(results[0])))
        w.writeheader()
        w.writerows(asdict(r) for r in results)
    print(f"wrote {csv_path}", file=sys.stderr)
    if args.plot:
        plot(results, args.out / f"{args.dataset}.png")


if __name__ == "__main__":
    main()
