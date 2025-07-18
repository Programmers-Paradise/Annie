import time, json, argparse, os, psutil, gc
import numpy as np
from rust_annie import AnnIndex, Distance
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
import faiss
import annoy

# Standardized datasets
DATASETS = {
    "small": {"N": 10_000, "D": 64, "k": 10},
    "medium": {"N": 100_000, "D": 128, "k": 20},
    "large": {"N": 1_000_000, "D": 256, "k": 50}
}

def measure_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # Return in MB

def benchmark(dataset="medium", repeats=50):
    config = DATASETS[dataset]
    N, D, k = config["N"], config["D"], config["k"]
    
    # Prepare data
    data = np.random.rand(N, D).astype(np.float32)
    ids = np.arange(N, dtype=np.int64)
    queries = np.random.rand(repeats, D).astype(np.float32)
    
    results = {
        "timestamp": datetime.utcnow().timestamp(),
        "dataset": dataset,
        "config": config,
        "repeats": repeats,
        "rust_annie": {},
        "sklearn": {},
        "faiss": {},
        "annoy": {}
    }

    # Rust benchmark
    gc.collect()
    mem_before = measure_memory()
    build_start = time.perf_counter()
    idx = AnnIndex(D, Distance.EUCLIDEAN)
    idx.add(data, ids)
    build_time = time.perf_counter() - build_start
    mem_after = measure_memory()
    
    search_times = []
    for q in queries:
        start = time.perf_counter()
        idx.search(q, k)
        search_times.append(time.perf_counter() - start)
    
    results["rust_annie"] = {
        "build_time": build_time,
        "build_memory_mb": mem_after - mem_before,
        "search_times": search_times,
        "search_avg": np.mean(search_times),
        "search_p50": np.percentile(search_times, 50),
        "search_p95": np.percentile(search_times, 95),
        "search_p99": np.percentile(search_times, 99)
    }

    # Scikit-learn benchmark
    try:
        gc.collect()
        mem_before = measure_memory()
        build_start = time.perf_counter()
        nn = NearestNeighbors(n_neighbors=k, algorithm='brute')
        nn.fit(data)
        build_time = time.perf_counter() - build_start
        mem_after = measure_memory()
        
        search_times = []
        for q in queries:
            start = time.perf_counter()
            nn.kneighbors(q.reshape(1, -1), return_distance=False)
            search_times.append(time.perf_counter() - start)
        
        results["sklearn"] = {
            "build_time": build_time,
            "build_memory_mb": mem_after - mem_before,
            "search_avg": np.mean(search_times),
            "search_p50": np.percentile(search_times, 50)
        }
    except ImportError:
        pass

    # FAISS benchmark
    try:
        gc.collect()
        mem_before = measure_memory()
        build_start = time.perf_counter()
        index = faiss.IndexFlatL2(D)
        index.add(data)
        build_time = time.perf_counter() - build_start
        mem_after = measure_memory()
        
        search_times = []
        for q in queries:
            start = time.perf_counter()
            _, _ = index.search(q.reshape(1, -1), k)
            search_times.append(time.perf_counter() - start)
        
        results["faiss"] = {
            "build_time": build_time,
            "build_memory_mb": mem_after - mem_before,
            "search_avg": np.mean(search_times),
            "search_p50": np.percentile(search_times, 50)
        }
    except ImportError:
        pass

    # Annoy benchmark
    try:
        gc.collect()
        mem_before = measure_memory()
        build_start = time.perf_counter()
        t = annoy.AnnoyIndex(D, 'euclidean')
        t.add_items(np.arange(N), data)
        t.build(10)
        build_time = time.perf_counter() - build_start
        mem_after = measure_memory()
        
        search_times = []
        for q in queries:
            start = time.perf_counter()
            t.get_nns_by_vector(q, k)
            search_times.append(time.perf_counter() - start)
        
        results["annoy"] = {
            "build_time": build_time,
            "build_memory_mb": mem_after - mem_before,
            "search_avg": np.mean(search_times),
            "search_p50": np.percentile(search_times, 50)
        }
    except ImportError:
        pass

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    results = benchmark(args.dataset, args.repeats)
    print(json.dumps(results, indent=2))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f)