"""Serve searches from many Python threads while another thread adds data."""

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from rust_annie import ThreadSafeAnnIndex

rng = np.random.default_rng(3)
index = ThreadSafeAnnIndex(16)
index.add(rng.standard_normal((10_000, 16)), np.arange(10_000))


def writer():
    for start in range(10_000, 12_000, 100):
        index.add(rng.standard_normal((100, 16)), np.arange(start, start + 100))


t = threading.Thread(target=writer)
t.start()
with ThreadPoolExecutor(max_workers=8) as pool:
    results = list(pool.map(lambda q: index.search(q, k=3), rng.standard_normal((200, 16))))
t.join()
print(f"{len(results)} searches done; index now holds {len(index)} vectors")
