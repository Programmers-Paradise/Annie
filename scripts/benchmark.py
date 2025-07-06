# benchmark.py
import time,json,argparse
import numpy as np
from rust_annie import AnnIndex, Distance
import os
from datetime import datetime

def pure_python_search(data, ids, q, k):
    # data: (N,D), q: (D,)
    # compute L2 distances and return top‐k
    dists = np.linalg.norm(data - q, axis=1)
    idx = np.argsort(dists)[:k]
    return ids[idx], dists[idx]

def benchmark(N=10000, D=64, k=10, repeats=50):
    # 1. Prepare random data
    data = np.random.rand(N, D).astype(np.float32)
    ids  = np.arange(N, dtype=np.int64)
    q    = data[0]

    # 2. Build Rust index
    idx = AnnIndex(D, Distance.EUCLIDEAN)
    idx.add(data, ids)
    # warm-up
    idx.search(q, k)

    # 3. Benchmark Rust search
    t0 = time.perf_counter()
    for _ in range(repeats):
        idx.search(q, k)
    t_rust = (time.perf_counter() - t0) / repeats

    # 4. Benchmark pure-Python search
    t0 = time.perf_counter()
    for _ in range(repeats):
        pure_python_search(data, ids, q, k)
    t_py = (time.perf_counter() - t0) / repeats

    results = {
        "rust_search_ms": t_rust * 1e3,
        "python_search_ms": t_py * 1e3,
        "speedup": t_py / t_rust
    }

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, help="Path to write benchmark results")
    args = parser.parse_args()

    results = benchmark()
    print(json.dumps(results, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f)
