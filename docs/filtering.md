## ANN Search Filtering

This document explains how to use the new filtering capabilities added in PR #338 to improve Approximate Nearest Neighbor (ANN) search.

### Why Filtering?

Filters allow you to narrow down search results dynamically based on:
- Metadata (e.g., tags, IDs, labels)
- Numeric thresholds (e.g., only items above/below a value)
- Custom user-defined logic

This improves both precision and flexibility of search.

#### Example: Python API

```python
from rust_annie import AnnIndex, PythonFilter
import numpy as np

# 1. Create an index with vector dimension 128
index = AnnIndex(dimension=128)

# 2. Add data with metadata
vector0 = np.random.rand(128).astype(np.float32)
vector1 = np.random.rand(128).astype(np.float32)

index.add_item(0, vector0, metadata={"category": "A"})
index.add_item(1, vector1, metadata={"category": "B"})

# 3. Create a filter (e.g., only include items where category == "A")
category_filter = PythonFilter.equals("category", "A")

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

### Integration & Extensibility

- Filters are exposed from Rust to Python via **PyO3** bindings.
- New filters can be added by extending `src/filters.rs` in the Rust code.
- Filters integrate cleanly with the existing ANN index search logic, so adding or combining filters doesn't require changes in the core search API.

### See also

- Example usage: [`scripts/filter_example.py`](scripts/filter_example.py)
- Unit tests covering filter behavior: [`tests/test_filters.py`](tests/test_filters.py)
