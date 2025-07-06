```markdown
## ANN Search Filtering

This document explains how to use the filtering capabilities to improve Approximate Nearest Neighbor (ANN) search.

### Why Filtering?

Filters allow you to narrow down search results dynamically based on:
- Metadata (e.g., tags, IDs, labels)
- Numeric thresholds (e.g., only items above/below a value)
- Custom user-defined logic

This improves both precision and flexibility of search.

#### Example: Python API

```python
from rust_annie import AnnIndex
import numpy as np

# 1. Create an index with vector dimension 128
index = AnnIndex(dimension=128)

# 2. Add data with metadata
vector0 = np.random.rand(128).astype(np.float32)
vector1 = np.random.rand(128).astype(np.float32)

index.add_item(0, vector0, metadata={"category": "A"})
index.add_item(1, vector1, metadata={"category": "B"})

# 3. Define a filter function (e.g., only include items where category == "A")
def category_filter(metadata):
    return metadata.get("category") == "A"

# 4. Perform search with the filter applied
query_vector = np.random.rand(128).astype(np.float32)
results = index.search(query_vector, k=5, filter=category_filter)

print("Filtered search results:", results)
```

### Supported Filters

This library supports applying filters to narrow down ANN search results dynamically.

| Filter type        | Example                                       |
|------------------- |----------------------------------------------- |
| **Equals**         | `Filter.equals("category", "A")`              |
| **Greater than**   | `Filter.gt("score", 0.8)`                     |
| **Less than**      | `Filter.lt("price", 100)`                     |
| **Custom predicate** | `Filter.custom(lambda metadata: ...)`       |

Filters work on the metadata you provide when adding items to the index.

### New Feature: Filtered Search with Custom Python Callbacks

The library now supports filtered search using custom Python callbacks, allowing for more complex filtering logic directly in Python.

#### Example: Filtered Search with Python Callback

```python
from rust_annie import AnnIndex, Distance
import numpy as np

# Create index
index = AnnIndex(3, Distance.EUCLIDEAN)
data = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
], dtype=np.float32)
ids = np.array([10, 20, 30], dtype=np.int64)
index.add(data, ids)

# Filter function
def even_ids(id: int) -> bool:
    return id % 2 == 0

# Filtered search
query = np.array([1.0, 2.0, 3.0], dtype=np.float32)
filtered_ids, filtered_dists = index.search_filter_py(
    query, 
    k=3, 
    filter_fn=even_ids
)
print(filtered_ids)  # [10, 30] (20 is filtered out)
```

### Sorting Behavior

The BruteForceIndex now uses `total_cmp` for sorting, which provides NaN-resistant sorting behavior. This change ensures that any NaN values in the data are handled consistently, preventing potential issues with partial comparisons.

### Benchmarking Indices

The library now includes a benchmarking function to evaluate the performance of different index types, specifically `PyHnswIndex` and `AnnIndex`. This function measures the average, maximum, and minimum query times, providing insights into the efficiency of each index type.

#### Example: Benchmarking Script

```python
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
    index.add(data, ids)

    # Warm-up + Timing
    times = []
    for i in range(q):
        start = time.perf_counter()
        _ = index.search(queries[i], k=k)
        times.append((time.perf_counter() - start) * 1000)

    print(f"  Avg query time: {np.mean(times):.3f} ms")
    print(f"  Max query time: {np.max(times):.3f} ms")
    print(f"  Min query time: {np.min(times):.3f} ms")

if __name__ == "__main__":
    benchmark(PyHnswIndex, "HNSW")
    benchmark(AnnIndex, "Brute-Force")
```

### Integration & Extensibility

- Filters are exposed from Rust to Python via **PyO3** bindings.
- New filters can be added by extending `src/filters.rs` in the Rust code.
- Filters integrate cleanly with the existing ANN index search logic, so adding or combining filters doesn't require changes in the core search API.

### See also

- Example usage: [`scripts/filter_example.py`](../scripts/filter_example.py)
- Unit tests covering filter behavior: [`tests/test_filters.py`](../tests/test_filters.py)
- Benchmarking script: [`scripts/compare.py`](../scripts/compare.py)
```