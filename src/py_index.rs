//! `rust_annie.Index`: one class that can use either backend.

use std::collections::HashMap;

use numpy::{AllowTypeChange, PyArrayLike1, PyArrayLike2};
use pyo3::prelude::*;

use crate::hnsw_index::{HnswConfig, HnswIndex};
use crate::index::{batch_to_numpy, to_numpy, AnnIndex};
use crate::metrics::Distance;
use crate::path_validation::validate_path_secure;
use crate::storage::{self, IndexKind};

enum Backend {
    Brute(AnnIndex),
    Hnsw(HnswIndex),
}

/// k-NN index with a selectable backend.
///
/// Parameters
/// ----------
/// kind : {"brute", "hnsw"}
///     ``"brute"`` for exact search, ``"hnsw"`` for approximate search.
/// dim : int
///     Dimensionality of the vectors.
/// metric : Distance, default Distance.EUCLIDEAN
#[pyclass(name = "Index", module = "rust_annie")]
pub struct PyIndex {
    backend: Backend,
}

#[pymethods]
impl PyIndex {
    #[new]
    #[pyo3(signature = (kind, dim, metric = Distance::Euclidean()))]
    fn new(kind: &str, dim: usize, metric: Distance) -> PyResult<Self> {
        let backend = match kind.to_lowercase().as_str() {
            "brute" => Backend::Brute(AnnIndex::create(dim, metric)?),
            "hnsw" => Backend::Hnsw(
                HnswIndex::new(dim, metric, HnswConfig::default())
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
            ),
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown backend '{}'. Use 'brute' or 'hnsw'.",
                    kind
                )))
            }
        };
        Ok(Self { backend })
    }

    /// Backend name: ``"brute"`` or ``"hnsw"``.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.backend {
            Backend::Brute(_) => "brute",
            Backend::Hnsw(_) => "hnsw",
        }
    }

    /// Add vectors with unique, non-negative integer IDs.
    fn add(
        &mut self,
        py: Python<'_>,
        data: PyArrayLike2<'_, f32, AllowTypeChange>,
        ids: PyArrayLike1<'_, i64, AllowTypeChange>,
    ) -> PyResult<()> {
        let ids = ids.as_array().to_vec();
        let view = data.as_array();
        match &mut self.backend {
            Backend::Brute(b) => py.allow_threads(|| b.add_rows(view, &ids)),
            Backend::Hnsw(h) => py.allow_threads(|| h.add_rows(view, &ids)),
        }
    }

    /// k-NN search for one query. Returns ``(ids, distances)``.
    #[pyo3(signature = (query, k, filter = None))]
    fn search(
        &self,
        py: Python<'_>,
        query: PyArrayLike1<'_, f32, AllowTypeChange>,
        k: usize,
        filter: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(PyObject, PyObject)> {
        let q = query.as_array().to_vec();
        let (ids, dists) = match &self.backend {
            Backend::Brute(b) => {
                let f = b.resolve_filter(filter)?;
                if b.metric.is_custom() {
                    b.search_vec(&q, k, f.as_ref())?
                } else {
                    py.allow_threads(|| b.search_vec(&q, k, f.as_ref()))?
                }
            }
            Backend::Hnsw(h) => {
                let f = h.resolve_filter(filter)?;
                py.allow_threads(|| h.search_vec(&q, k, None, f.as_ref()))?
            }
        };
        Ok(to_numpy(py, ids, dists))
    }

    /// k-NN search for each row of ``queries``; returns ``(n, k)`` arrays.
    #[pyo3(signature = (queries, k, filter = None))]
    fn search_batch(
        &self,
        py: Python<'_>,
        queries: PyArrayLike2<'_, f32, AllowTypeChange>,
        k: usize,
        filter: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(PyObject, PyObject)> {
        let view = queries.as_array();
        let n = view.nrows();
        let (ids, dists) = match &self.backend {
            Backend::Brute(b) => {
                let f = b.resolve_filter(filter)?;
                if b.metric.is_custom() {
                    b.search_batch_vec(view, k, f.as_ref())?
                } else {
                    py.allow_threads(|| b.search_batch_vec(view, k, f.as_ref()))?
                }
            }
            Backend::Hnsw(h) => {
                let f = h.resolve_filter(filter)?;
                py.allow_threads(|| h.search_batch_vec(view, k, None, f.as_ref()))?
            }
        };
        batch_to_numpy(py, n, k, ids, dists)
    }

    /// Remove vectors by ID. Raises ``KeyError`` if any ID is missing.
    fn remove(&mut self, ids: Vec<i64>) -> PyResult<()> {
        match &mut self.backend {
            Backend::Brute(b) => b.remove_ids(&ids),
            Backend::Hnsw(h) => h.remove_ids(&ids),
        }
    }

    /// Save to ``path``.
    fn save(&self, path: &str) -> PyResult<()> {
        let p = validate_path_secure(path)?;
        match &self.backend {
            Backend::Brute(b) => b.save_to(&p),
            Backend::Hnsw(h) => h.save_to(&p),
        }
    }

    /// Load an index of either kind written by ``save``.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let p = validate_path_secure(path)?;
        let backend = match storage::detect_kind(&p).map_err(|e| e.into_pyerr())? {
            IndexKind::BruteForce => Backend::Brute(AnnIndex::load_from(&p)?),
            IndexKind::Hnsw => Backend::Hnsw(HnswIndex::load_from(&p)?),
        };
        Ok(Self { backend })
    }

    fn __len__(&self) -> usize {
        match &self.backend {
            Backend::Brute(b) => b.size(),
            Backend::Hnsw(h) => h.size(),
        }
    }

    fn get_info(&self) -> HashMap<String, String> {
        match &self.backend {
            Backend::Brute(b) => b.info(),
            Backend::Hnsw(h) => h.get_info(),
        }
    }

    /// Check internal consistency; raises ``ValueError`` describing any issues.
    fn validate(&self) -> PyResult<()> {
        match &self.backend {
            Backend::Brute(b) => b.validate_integrity(),
            Backend::Hnsw(h) => h
                .validate()
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string())),
        }
    }

    fn __repr__(&self) -> String {
        let info = self.get_info();
        format!(
            "Index(kind={:?}, dim={}, metric={}, size={})",
            self.kind(),
            info["dim"],
            info["metric"],
            info["size"]
        )
    }
}
