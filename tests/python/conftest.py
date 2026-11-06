import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def dataset(rng):
    """500 random 16-d vectors with non-contiguous IDs, plus 20 queries."""
    data = rng.standard_normal((500, 16)).astype(np.float32)
    ids = np.arange(1000, 1500, dtype=np.int64)
    queries = rng.standard_normal((20, 16)).astype(np.float32)
    return data, ids, queries
