# batch_benchmark.py

import time,json,argparse
import numpy as np
from rust_annie import AnnIndex, Distance

def benchmark_batch(N=10000, D=64, k=10, batch_size=64, repeats=20):
    # 1. Prepare random data
    data = np.random.rand(N, D).astype(np.float32)
    ids  = np.arange(N, dtype=np.int64)
    idx  = AnnIndex(D, Distance.EUCLIDEAN)
    idx.add(data, ids)

    # 2. Prepare query batch
    queries = data[:batch_size]

    # Warm-up
    idx.search_batch(queries, k)

    # 3. Benchmark Rust batch search
    t0 = time.perf_counter()
    for _ in range(repeats):
        idx.search_batch(queries, k)
    t_batch = (time.perf_counter() - t0) / repeats

    results = {
        "batch_time_ms": t_batch * 1e3,
        "per_query_time_ms": (t_batch / batch_size) * 1e3
    }

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, help="Path to write benchmark results")
    args = parser.parse_args()

    results = benchmark_batch()
    print(json.dumps(results, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f)

