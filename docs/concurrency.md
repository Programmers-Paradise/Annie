# Using `ThreadSafeAnnIndex` for Concurrent Access

Annie exposes a thread-safe version of its ANN index (`AnnIndex`) for use in Python. This is useful when you want to perform parallel search or update operations from Python threads.

## Key Features

- Safe concurrent read access (`search`, `search_batch`)
- Exclusive write access (`add`, `remove`)
- Backed by Rust `RwLock` and exposed via PyO3

## Example

```python
from annie import ThreadSafeAnnIndex, Distance
import numpy as np
import threading

# Create index
index = ThreadSafeAnnIndex(128, Distance.Cosine)

# Add vectors
data = np.random.rand(1000, 128).astype('float32')
ids = np.arange(1000, dtype=np.int64)
index.add(data, ids)

# Run concurrent searches
def run_search():
    query = np.random.rand(128).astype('float32')
    ids, distances = index.search(query, 10)
    print(ids)

threads = [threading.Thread(target=run_search) for _ in range(4)]
[t.start() for t in threads]
[t.join() for t in threads]
```
