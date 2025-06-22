# Rust-annie

![Annie](https://github.com/Programmers-Paradise/.github/blob/main/ChatGPT%20Image%20May%2015,%202025,%2003_58_16%20PM.png?raw=true)

[![PyPI](https://img.shields.io/pypi/v/rust-annie.svg)](https://pypi.org/project/rust-annie)  


A lightning-fast, Rust-powered brute-force k-NN library for Python, with optional batch queries, thread-safety, and on-disk persistence.

---

## 📝 Table of Contents

1. [Features](#features)  
2. [Installation](#installation)  
3. [Quick Start](#quick-start)  
4. [Examples](#examples)  
   - [Single Query](#single-query)  
   - [Batch Query](#batch-query)  
   - [Thread-Safe Usage](#thread-safe-usage)  
5. [Benchmark Results](#benchmark-results)  
6. [API Reference](#api-reference)  
7. [Development & CI](#development--ci)  
8. [Roadmap](#roadmap)  
9. [Contributing](#contributing)  
10. [License](#license)  

---

## 🚀 Features

- **Ultra-fast brute-force** k-NN search (Euclidean , Cosine, Manhattan)  
- **Batch** queries over multiple vectors  
- **Thread-safe** wrapper with GIL release for true concurrency  
- **Zero-copy** NumPy integration (via PyO3 & rust-numpy)  
- **On-disk** persistence with bincode + serde  
- **Multi-platform** wheels (manylinux, musllinux, Windows, macOS)  
- **Automated CI** with correctness & performance checks  

---

## ⚙️ Installation

```bash
# Stable release from PyPI:
pip install rust-annie

# Or install from source (requires Rust toolchain + maturin):
git clone https://github.com/yourusername/rust_annie.git
cd rust_annie
pip install maturin
maturin develop --release
```




## 🎉 Quick Start

```python
import numpy as np
from rust_annie import AnnIndex, Distance

# Create an 8-dim Euclidean index
idx = AnnIndex(8, Distance.EUCLIDEAN)

# Add 100 random vectors
data = np.random.rand(100, 8).astype(np.float32)
ids  = np.arange(100, dtype=np.int64)
idx.add(data, ids)

# Query one vector
labels, dists = idx.search(data[0], k=5)
print("Nearest IDs:", labels)
print("Distances :", dists)
```

---

## 📚 Examples

### Single Query

```python
from rust_annie import AnnIndex, Distance
import numpy as np

idx = AnnIndex(4, Distance.COSINE)
data = np.random.rand(50, 4).astype(np.float32)
ids  = np.arange(50, dtype=np.int64)
idx.add(data, ids)

labels, dists = idx.search(data[10], k=3)
print(labels, dists)
```

### Batch Query

```python
from rust_annie import AnnIndex, Distance
import numpy as np

idx = AnnIndex(16, Distance.EUCLIDEAN)
data = np.random.rand(1000, 16).astype(np.float32)
ids  = np.arange(1000, dtype=np.int64)
idx.add(data, ids)

# Query 32 vectors at once:
queries = data[:32]
labels_batch, dists_batch = idx.search_batch(queries, k=10)
print(labels_batch.shape)  # (32, 10)
```

### Thread-Safe Usage

```python
from rust_annie import ThreadSafeAnnIndex, Distance
import numpy as np
from concurrent.futures import ThreadPoolExecutor

idx = ThreadSafeAnnIndex(32, Distance.EUCLIDEAN)
data = np.random.rand(500, 32).astype(np.float32)
ids  = np.arange(500, dtype=np.int64)
idx.add(data, ids)

def task(q):
    return idx.search(q, k=5)

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(task, data[i]) for i in range(8)]
    for f in futures:
        print(f.result())
```

---

## Build and Query a Brute-Force AnnIndex in Python (Complete Example)

This section demonstrates a complete, beginner-friendly example of how to build and query a `brute-force AnnIndex` using Python.

> A brute-force AnnIndex exhaustively compares the query vector with every vector in the dataset. Though it checks all vectors, it's **extremely fast** thanks to its underlying **Rust + SIMD** implementation.

---

## Steps

- Initialize a `brute-force AnnIndex` with 128 dimensions and cosine distance.
- Generate and add a batch of random vectors with unique IDs.
- Perform a top-5 nearest-neighbor search on a new query vector.
- Print the IDs and distances of the closest matches.


---

### 💻 Code Example

> Make sure you’ve installed the library first:

```bash
pip install rust-annie  # if not installed already
```

```python
import numpy as np
from rust_annie import AnnIndex, Distance

index = AnnIndex(dim=128, metric=Distance.COSINE)

vectors = np.random.rand(1000, 128).astype(np.float32)

ids = np.arange(1000, dtype=np.int64)

index.add(vectors, ids)

query = np.random.rand(128).astype(np.float32)

top_ids, distances = index.search(query, k=5)

print("Top 5 nearest neighbors:")

for i in range(5):
    print(f"ID: {top_ids[i]}, Distance: {distances[i]}")
```


## 📈 Benchmark Results

Measured on a 6-core CPU:

| Mode                             | Per-query Time |
| -------------------------------- | -------------: |
| Pure-Python (NumPy - 𝑙2)        |       \~2.8 ms |
| Rust AnnIndex single query       |       \~0.7 ms |
| Rust AnnIndex batch (64 queries) |      \~0.23 ms |

That’s a \~4× speedup vs. NumPy!

---

## 📖 API Reference

### `rust_annie.AnnIndex(dim: int, metric: Distance)`

Create a new brute-force index.

### Methods

* `add(data: np.ndarray[N×D], ids: np.ndarray[N]) -> None`
* `search(query: np.ndarray[D], k: int) -> (ids: np.ndarray[k], dists: np.ndarray[k])`
* `search_batch(data: np.ndarray[N×D], k: int) -> (ids: np.ndarray[N×k], dists: np.ndarray[N×k])`
* `remove(ids: Sequence[int]) -> None`
* `save(path: str) -> None`
* `load(path: str) -> AnnIndex` (static)

### `rust_annie.Distance`

Enum: `Distance.EUCLIDEAN`, `Distance.COSINE`.`Distance.MANHATTAN`

### `rust_annie.ThreadSafeAnnIndex`

Same API as `AnnIndex`, safe for concurrent use.

---

## 🔧 Development & CI

**CI** runs on GitHub Actions, building wheels on Linux, Windows, macOS, plus:

* `cargo test`
* `pytest`
* `benchmark.py` & `batch_benchmark.py`

```bash
# Locally run tests & benchmarks
cargo test
pytest
python benchmark.py
python batch_benchmark.py
```

---

## 🚧 Roadmap

* [x] SIMD-accelerated dot products
* [x] Rayon parallelism & GIL release
* [ ] Integrate HNSW/FAISS for sub-ms ANN at scale
* [ ] GPU‐backed search (CUDA/ROCm)
* [ ] Richer Python docs & type hints

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repo
2. Create a feature branch
3. Add tests & docs
4. Submit a Pull Request

See [CONTRIBUTING.md](./CONTRIBUTING.md) for details.

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](./LICENSE) for details.

```
```
