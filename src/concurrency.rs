//! Thread-safe Python wrapper around [`AnnIndex`].
//!
//! Locks are always acquired with the GIL released, and the GIL is only
//! re-acquired (for custom metrics or callable filters) while holding a lock.
//! This ordering rules out GIL/lock deadlocks between Python threads.

use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use numpy::{AllowTypeChange, PyArrayLike1, PyArrayLike2};
use pyo3::prelude::*;

use crate::filters::Filter;
use crate::index::{batch_to_numpy, to_numpy, AnnIndex};
use crate::metrics::Distance;
use crate::path_validation::validate_path_secure;

fn read(lock: &RwLock<AnnIndex>) -> PyResult<RwLockReadGuard<'_, AnnIndex>> {
    lock.read().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Index lock poisoned: {}", e))
    })
}

fn write(lock: &RwLock<AnnIndex>) -> PyResult<RwLockWriteGuard<'_, AnnIndex>> {
    lock.write().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Index lock poisoned: {}", e))
    })
}

/// Thread-safe exact k-NN index.
///
/// Same API as `AnnIndex`, but safe to share between Python threads: any
/// number of concurrent searches may run alongside each other, while
/// mutations (`add`, `remove`, `update`, `compact`) take an exclusive lock.
#[pyclass(module = "rust_annie")]
pub struct ThreadSafeAnnIndex {
    inner: Arc<RwLock<AnnIndex>>,
}

/// Convert an optional Python filter into a `Send`-able owned object.
fn own_filter(filter: Option<&Bound<'_, PyAny>>) -> Option<Py<PyAny>> {
    filter.filter(|f| !f.is_none()).map(|f| f.clone().unbind())
}

/// Resolve a filter while holding the index lock (needs the GIL for callables).
fn resolve(idx: &AnnIndex, filter: &Option<Py<PyAny>>) -> PyResult<Option<Filter>> {
    match filter {
        None => Ok(None),
        Some(obj) => Python::with_gil(|py| idx.resolve_filter(Some(obj.bind(py)))),
    }
}

#[pymethods]
impl ThreadSafeAnnIndex {
    #[new]
    #[pyo3(signature = (dim, metric = Distance::Euclidean()))]
    pub fn new(dim: usize, metric: Distance) -> PyResult<Self> {
        Ok(ThreadSafeAnnIndex {
            inner: Arc::new(RwLock::new(AnnIndex::create(dim, metric)?)),
        })
    }

    /// Add vectors with unique, non-negative integer IDs.
    pub fn add(
        &self,
        py: Python<'_>,
        data: PyArrayLike2<'_, f32, AllowTypeChange>,
        ids: PyArrayLike1<'_, i64, AllowTypeChange>,
    ) -> PyResult<()> {
        let ids = ids.as_array().to_vec();
        let view = data.as_array();
        let lock = &self.inner;
        py.allow_threads(|| write(lock)?.add_rows(view, &ids))
    }

    /// Remove vectors by ID. Raises `KeyError` if any ID is missing.
    pub fn remove(&self, py: Python<'_>, ids: Vec<i64>) -> PyResult<()> {
        let lock = &self.inner;
        py.allow_threads(|| write(lock)?.remove_ids(&ids))
    }

    /// Replace the vector stored under `id`.
    pub fn update(
        &self,
        py: Python<'_>,
        id: i64,
        vector: PyArrayLike1<'_, f32, AllowTypeChange>,
    ) -> PyResult<()> {
        let v = vector.as_array().to_vec();
        let lock = &self.inner;
        py.allow_threads(|| write(lock)?.update_vector(id, &v))
    }

    /// Reclaim storage left by removed entries.
    pub fn compact(&self, py: Python<'_>) -> PyResult<()> {
        let lock = &self.inner;
        py.allow_threads(|| {
            write(lock)?.compact_entries();
            Ok(())
        })
    }

    /// Exact k-NN search for one query; see `AnnIndex.search`.
    #[pyo3(signature = (query, k, filter = None))]
    pub fn search(
        &self,
        py: Python<'_>,
        query: PyArrayLike1<'_, f32, AllowTypeChange>,
        k: usize,
        filter: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(PyObject, PyObject)> {
        let q = query.as_array().to_vec();
        let filter = own_filter(filter);
        let lock = &self.inner;
        let (ids, dists) = py.allow_threads(|| {
            let idx = read(lock)?;
            let f = resolve(&idx, &filter)?;
            idx.search_vec(&q, k, f.as_ref())
        })?;
        Ok(to_numpy(py, ids, dists))
    }

    /// Exact k-NN search for a batch of queries; see `AnnIndex.search_batch`.
    #[pyo3(signature = (queries, k, filter = None))]
    pub fn search_batch(
        &self,
        py: Python<'_>,
        queries: PyArrayLike2<'_, f32, AllowTypeChange>,
        k: usize,
        filter: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(PyObject, PyObject)> {
        let view = queries.as_array();
        let n = view.nrows();
        let filter = own_filter(filter);
        let lock = &self.inner;
        let (ids, dists) = py.allow_threads(|| {
            let idx = read(lock)?;
            let f = resolve(&idx, &filter)?;
            idx.search_batch_vec(view, k, f.as_ref())
        })?;
        batch_to_numpy(py, n, k, ids, dists)
    }

    /// Save to `path`.
    pub fn save(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        let p = validate_path_secure(path)?;
        let lock = &self.inner;
        py.allow_threads(|| read(lock)?.save_to(&p))
    }

    /// Load an `AnnIndex` file and wrap it.
    #[staticmethod]
    pub fn load(path: &str) -> PyResult<Self> {
        let p = validate_path_secure(path)?;
        Ok(Self::from_index(AnnIndex::load_from(&p)?))
    }

    pub fn version(&self, py: Python<'_>) -> PyResult<u64> {
        let lock = &self.inner;
        py.allow_threads(|| {
            Ok(read(lock)?
                .version
                .load(std::sync::atomic::Ordering::Relaxed))
        })
    }

    pub fn len(&self, py: Python<'_>) -> PyResult<usize> {
        let lock = &self.inner;
        py.allow_threads(|| Ok(read(lock)?.size()))
    }

    pub fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        self.len(py)
    }

    pub fn __contains__(&self, py: Python<'_>, id: i64) -> PyResult<bool> {
        let lock = &self.inner;
        py.allow_threads(|| Ok(read(lock)?.slots.contains_key(&id)))
    }

    pub fn dim(&self, py: Python<'_>) -> PyResult<usize> {
        let lock = &self.inner;
        py.allow_threads(|| Ok(read(lock)?.dim))
    }

    pub fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let lock = &self.inner;
        py.allow_threads(|| {
            let idx = read(lock)?;
            Ok(format!(
                "ThreadSafeAnnIndex(dim={}, metric={}, size={})",
                idx.dim,
                idx.metric.name(),
                idx.size()
            ))
        })
    }
}

impl ThreadSafeAnnIndex {
    /// Wrap an existing index.
    pub fn from_index(idx: AnnIndex) -> Self {
        Self {
            inner: Arc::new(RwLock::new(idx)),
        }
    }

    /// Wrap an existing shared lock (used in tests).
    pub fn from_arc(inner: Arc<RwLock<AnnIndex>>) -> Self {
        Self { inner }
    }
}
