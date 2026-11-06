# Benchmarks

`benchmarks/compare.py` compares rust-annie with other k-NN libraries using the
[ANN-Benchmarks](https://ann-benchmarks.com) methodology: standard datasets with
precomputed ground truth, recall@k, and queries issued one at a time on a single thread.
It also reports batch throughput using all cores, and index build time.

Libraries compared (each is skipped if not installed):

| Library | Exact | Approximate |
|---|---|---|
| rust-annie | `AnnIndex` | `PyHnswIndex` |
| [FAISS](https://github.com/facebookresearch/faiss) | `IndexFlatL2` | `IndexHNSWFlat` |
| [hnswlib](https://github.com/nmslib/hnswlib) | – | `hnswlib.Index` |
| NumPy | matrix product + `argpartition` | – |

All HNSW implementations use the same `M` and `ef_construction` and are swept over the
same `ef` values. Cosine datasets are L2-normalised for FAISS.

## Reproducing

```bash
pip install rust-annie hnswlib faiss-cpu h5py matplotlib

# quick synthetic check (10 000 x 32, no download)
python benchmarks/compare.py --dataset random-xs

# standard datasets (downloaded once to ~/.cache/rust_annie_bench)
python benchmarks/compare.py --dataset sift-128-euclidean --plot
python benchmarks/compare.py --dataset glove-100-angular --plot
```

Results are written to `benchmarks/results/<dataset>.csv` (first line records the
machine) and, with `--plot`, a recall/QPS plot. Run `--help` for all options
(`--m`, `--ef-construction`, `--ef`, `--k`, `--max-queries`, `--libraries`).

For Rust-level micro-benchmarks run `cargo bench`.

## Results

Results depend strongly on hardware; always report the machine.

`random-xs` (10 000 × 32 Gaussian vectors, Euclidean, 200 queries, k = 10), rust-annie
0.3.0, Linux x86-64, 8 CPUs. hnswlib and FAISS were not installed for this run.

| Library | Method | Parameters | Recall@10 | QPS (1 thread) | QPS (batch) | Build (s) |
|---|---|---|---|---|---|---|
| rust-annie | exact | – | 1.000 | 2 123 | 6 781 | 0.00 |
| rust-annie | HNSW | M=16, efC=200, ef=20 | 0.831 | 4 341 | 26 479 | 2.19 |
| rust-annie | HNSW | M=16, efC=200, ef=80 | 0.984 | 1 553 | 8 575 | 2.19 |
| rust-annie | HNSW | M=16, efC=200, ef=320 | 1.000 | 507 | 2 838 | 2.19 |
| NumPy | exact (BLAS) | – | 1.000 | 5 470 | 5 527 | – |

At this small size, single exact queries are faster with NumPy's BLAS matrix–vector
product than with rust-annie's exact search, and HNSW only pays off at low `ef`;
approximate indexes are intended for much larger collections. Results on the standard
SIFT-1M and GloVe-100 datasets, including hnswlib and FAISS, are generated with the
commands above.
