import time, json, argparse
import numpy as np
from rust_annie import AnnIndex, Distance


def benchmark_batch(N=10000, D=64, k=10, batch_size=64, repeats=20):
    # 1. Prepare random data
    data = np.random.rand(N, D).astype(np.float32)
    ids = np.arange(N, dtype=np.int64)
    idx = AnnIndex(D, Distance.EUCLIDEAN)
    idx.add(data, ids)

    # 2. Prepare query batch
    queries = data[np.random.randint(low=0,high=N,size=(batch_size),dtype=np.int64)]

    # Warm-up
    idx.search_batch(queries, k, None)  # Added None for filter

    # 3. Benchmark Rust batch search
    t_total = 0
    for _ in range(repeats):
        queries = data[np.random.randint(low=0,high=N,size=(batch_size),dtype=np.int64)] # takes a set of new random query every time
        t_start = time.perf_counter()
        idx.search_batch(queries, k, None)  # Added None for filter
        t_end = time.perf_counter()
        t_total += t_end - t_start
    t_batch = t_total / repeats

    results = {
        "batch_time_ms": t_batch * 1e3,
        "per_query_time_ms": (t_batch / batch_size) * 1e3,
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
