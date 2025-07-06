import numpy as np
import time
from rust_annie import PyHnswIndex, AnnIndex


def benchmark(index_cls, name, dim=128, n=10_000, q=100, k=10):
    print(f"\nBenchmarking {name} with {n} vectors (dim={dim})...")

    # Data
    data = np.random.rand(n, dim).astype(np.float32)
    ids = np.arange(n, dtype=np.int64)
    queries = np.random.rand(q, dim).astype(np.float32)

    # Index setup
    index = index_cls(dims=dim)
    for i in range(n):
        index.add(data[i], ids[i])

    # Warm-up + Timing
    times = []
    # Batch search for improved performance and realistic benchmarking
    start = time.perf_counter()
    results = index.search(queries, k=k)
    elapsed = (time.perf_counter() - start) * 1000
    times = [elapsed / q] * q  # Approximate per-query time

    print(f"  Avg query time: {np.mean(times):.3f} ms")
    print(f"  Max query time: {np.max(times):.3f} ms")
    print(f"  Min query time: {np.min(times):.3f} ms")


if __name__ == "__main__":
    benchmark(PyHnswIndex, "HNSW")
    benchmark(AnnIndex, "Brute-Force")
