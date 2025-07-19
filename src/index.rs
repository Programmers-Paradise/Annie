use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, IntoPyArray};
use ndarray::Array2;
use ndarray::s;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use std::time::Instant;

use crate::backend::AnnBackend;
use crate::storage::{save_index, load_index};
use crate::metrics::Distance;
use crate::errors::RustAnnError;
use crate::monitoring::MetricsCollector;

#[pyclass]
#[derive(Serialize, Deserialize)]
/// A brute-force k-NN index with cached norms, Rayon parallelism,
/// and support for L1, L2, Cosine, Chebyshev, and Minkowski-p distances.
pub struct AnnIndex {
    dim: usize,
    metric: Distance,
    /// If Some(p), use Minkowski-p distance instead of `metric`.
    minkowski_p: Option<f32>,
    /// Stored entries as (id, vector, squared_norm) tuples.
    entries: Vec<(i64, Vec<f32>, f32)>,
    /// Optional metrics collector for monitoring
    #[serde(skip)]
    metrics: Option<Arc<MetricsCollector>>,
}

#[pymethods]
impl AnnIndex {
    #[new]
    /// Create a new index for unit-variant metrics.
    pub fn new(dim: usize, metric: Distance) -> PyResult<Self> {
        if dim == 0 {
            return Err(RustAnnError::py_err("Invalid Dimension", "Dimension must be > 0"));
        }
        Ok(AnnIndex { dim, metric, minkowski_p: None, entries: Vec::new(), metrics: None })
    }

    #[staticmethod]
    /// Create a new Minkowski-p index.
    pub fn new_minkowski(dim: usize, p: f32) -> PyResult<Self> {
        if dim == 0 {
            return Err(RustAnnError::py_err("Invalid Dimension", "Dimension must be > 0"));
        }
        if p <= 0.0 {
            return Err(RustAnnError::py_err("Minkowski Error", "`p` must be > 0 for Minkowski distance"));
        }
        Ok(AnnIndex { dim, metric: Distance::Minkowski(p), minkowski_p: Some(p), entries: Vec::new(), metrics: None })
    }

    #[staticmethod]
    /// Create a new index with a named metric.
    pub fn new_with_metric(dim: usize, metric_name: &str) -> PyResult<Self> {
        if dim == 0 {
            return Err(RustAnnError::py_err("Invalid Dimension", "Dimension must be > 0"));
        }
        let metric = Distance::new(metric_name);
        Ok(AnnIndex { dim, metric, minkowski_p: None, entries: Vec::new(), metrics: None })
    }

    /// Add vectors and IDs in batch.
    pub fn add(&mut self, _py: Python, data: PyReadonlyArray2<f32>, ids: PyReadonlyArray1<i64>) -> PyResult<()> {
        self.add_batch_internal(data, ids, None)
    }

    /// Add vectors and IDs in batch with progress reporting.
    /// The callback should be a callable that takes two integers: (current, total)
    pub fn add_batch_with_progress(
        &mut self,
        py: Python,
        data: PyReadonlyArray2<f32>,
        ids: PyReadonlyArray1<i64>,
        progress_callback: PyObject
    ) -> PyResult<()> {
        self.add_batch_internal(data, ids, Some(progress_callback))
    }

    fn add_batch_internal(
        &mut self,
        data: PyReadonlyArray2<f32>,
        ids: PyReadonlyArray1<i64>,
        progress_callback: Option<PyObject>
    ) -> PyResult<()> {
        let view = data.as_array();
        let ids = ids.as_slice()?;
        let n = view.nrows();
        if n != ids.len() {
            return Err(RustAnnError::py_err("Input Mismatch", "`data` and `ids` must have same length"));
        }
        if view.ncols() != self.dim {
            return Err(RustAnnError::py_err("Dimension Error", format!("Expected dimension {}, got {}", self.dim, view.ncols())));
        }
        
        self.entries.reserve(n);
        let chunk_size = 1000; // vectors per chunk
        let num_chunks = (n + chunk_size - 1) / chunk_size;
        
        for idx in 0..num_chunks {
            let start = idx * chunk_size;
            let end = (start + chunk_size).min(n);
            let chunk_view = view.slice(s![start..end, ..]);
            let chunk_ids = &ids[start..end];
            
            let mut seen_ids = std::collections::HashSet::new();
            new_entries = chunk_view.outer_iter()
                .zip(chunk_ids)
                .par_bridge()
                .map(|(row, &id)| {
                    let v = row.to_vec();
                    let sq_norm = v.iter().map(|x| x * x).sum::<f32>();
                    (id, v, sq_norm)
                })
                .collect();
            self.entries.extend(new_entries);
            
            if let Some(cb) = &progress_callback {
                Python::with_gil(|py| {
                    cb.call1(py, (end, n)).map_err(|e| {
                        RustAnnError::py_err("Callback Error", format!("Progress callback failed: {}", e))
                    })?;
                    Ok(())
                })?;
            }
        }
        
        if let Some(metrics) = &self.metrics {
            let metric_name = match &self.metric {
                    Distance::Euclidean() => "euclidean".to_string(),
                    Distance::Cosine()    => "cosine".to_string(),
                    Distance::Manhattan() => "manhattan".to_string(),
                    Distance::Chebyshev() => "chebyshev".to_string(),
                    Distance::Minkowski(p) => format!("minkowski_p{}", p),
                    Distance::Hamming()   => "hamming".to_string(),
                    Distance::Jaccard()   => "jaccard".to_string(),
                    Distance::Angular()   => "angular".to_string(),
                    Distance::Canberra()  => "canberra".to_string(),
                    Distance::Custom(n)   => n.clone(),
            };
            metrics.set_index_metadata(self.entries.len(), self.dim, &metric_name);
        }
        Ok(())
    }

    /// Remove entries by ID.
    pub fn remove(&mut self, ids: Vec<i64>) -> PyResult<()> {
        if !ids.is_empty() {
            let to_rm: std::collections::HashSet<i64> = ids.into_iter().collect();
            self.entries.retain(|(id, _, _)| !to_rm.contains(id));
        }
        Ok(())
    }

    /// Single query search.
    pub fn search(&self, py: Python, query: PyReadonlyArray1<f32>, k: usize) -> PyResult<(PyObject, PyObject)> {
        if self.entries.is_empty() {
            return Err(RustAnnError::py_err("EmptyIndex", "Index is empty"));
        }
        let q = query.as_slice()?;
        let q_sq = q.iter().map(|x| x * x).sum::<f32>();
        let start = Instant::now();
        let (ids, dists) = py.allow_threads(|| self.inner_search(q, q_sq, k))?;
        if let Some(metrics) = &self.metrics {
            metrics.record_query(start.elapsed());
        }
        Ok((ids.into_pyarray(py).into(), dists.into_pyarray(py).into()))
    }

    /// Batch queries search.
    pub fn search_batch(&self, py: Python, data: PyReadonlyArray2<f32>, k: usize) -> PyResult<(PyObject, PyObject)> {
        let arr = data.as_array();
        let n = arr.nrows();
        if arr.ncols() != self.dim {
            return Err(RustAnnError::py_err("Dimension Error", format!("Expected shape (N, {}), got (N, {})", self.dim, arr.ncols())));
        }
        let results: Result<Vec<_>, RustAnnError> = py.allow_threads(|| {
            (0..n).into_par_iter().map(|i| {
                let row = arr.row(i).to_vec();
                let q_sq = row.iter().map(|x| x * x).sum::<f32>();
                self.inner_search(&row, q_sq, k).map_err(|e| RustAnnError::io_err(format!("Parallel search failed: {}", e)))
            }).collect()
        });
        let results = results.map_err(|e| e.into_pyerr())?;
        let (all_ids, all_dists): (Vec<_>, Vec<_>) = results.into_iter().unzip();
        let ids_arr = Array2::from_shape_vec((n, k), all_ids.concat()).map_err(|e| RustAnnError::py_err("Reshape Error", format!("Reshape ids failed: {}", e)))?;
        let dists_arr = Array2::from_shape_vec((n, k), all_dists.concat()).map_err(|e| RustAnnError::py_err("Reshape Error", format!("Reshape dists failed: {}", e)))?;
        Ok((ids_arr.into_pyarray(py).into(), dists_arr.into_pyarray(py).into()))
    }

    /// Save index to file (.bin appended).
    pub fn save(&self, path: &str) -> PyResult<()> {
        Self::validate_path(path)?;
        let full = format!("{}.bin", path);
        save_index(self, &full).map_err(|e| e.into_pyerr())
    }

    #[staticmethod]
    /// Load index from file (.bin appended).
    pub fn load(path: &str) -> PyResult<Self> {
        Self::validate_path(path)?;
        let full = format!("{}.bin", path);
        load_index(&full).map_err(|e| e.into_pyerr())
    }

    /// Number of entries.
    pub fn len(&self) -> usize { self.entries.len() }
    pub fn __len__(&self) -> usize { self.entries.len() }

    /// Vector dimension.
    pub fn dim(&self) -> usize { self.dim }

    /// String repr.
    pub fn __repr__(&self) -> String {
        let m = if let Some(p) = self.minkowski_p { format!("Minkowski(p={})", p) } else { format!("{:?}", self.metric) };
        format!("AnnIndex(dim={}, metric={}, entries={})", self.dim, m, self.entries.len())
    }

    /// Enable metrics on optional port.
    pub fn enable_metrics(&mut self, port: Option<u16>) -> PyResult<()> {
        let metrics = Arc::new(MetricsCollector::new());
        let metric_name = if let Some(p) = self.minkowski_p { format!("minkowski_p{}", p) } else {
            match &self.metric {
                Distance::Euclidean() => "euclidean".to_string(),
                Distance::Cosine()    => "cosine".to_string(),
                Distance::Manhattan() => "manhattan".to_string(),
                Distance::Chebyshev() => "chebyshev".to_string(),
                Distance::Minkowski(p) => format!("minkowski_p{}", p),
                Distance::Hamming()   => "hamming".to_string(),
                Distance::Jaccard()   => "jaccard".to_string(),
                Distance::Angular()   => "angular".to_string(),
                Distance::Canberra()  => "canberra".to_string(),
                Distance::Custom(n)   => n.clone(),
            }
        };
        metrics.set_index_metadata(self.entries.len(), self.dim, &metric_name);
        if let Some(p) = port {
            use crate::monitoring::MetricsServer;
            let server = MetricsServer::new(Arc::clone(&metrics), p);
            server.start().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to start metrics server: {}", e)))?;
        }
        self.metrics = Some(metrics);
        Ok(())
    }

    /// Fetch metrics snapshot.
    pub fn get_metrics(&self) -> PyResult<Option<PyObject>> {
        if let Some(metrics) = &self.metrics {
            let snap = metrics.get_metrics_snapshot();
            Python::with_gil(|py| {
                let d = pyo3::types::PyDict::new(py);
                d.set_item("query_count", snap.query_count)?;
                d.set_item("avg_query_latency_us", snap.avg_query_latency_us)?;
                d.set_item("index_size", snap.index_size)?;
                d.set_item("dimensions", snap.dimensions)?;
                d.set_item("uptime_seconds", snap.uptime_seconds)?;
                d.set_item("distance_metric", snap.distance_metric)?;
                let recall = pyo3::types::PyDict::new(py);
                d.set_item("recall_estimates", recall)?;
                Ok(Some(d.into()))
            })
        } else {
            Ok(None)
        }
    }

    /// Update recall for k.
    pub fn update_recall_estimate(&self, k: usize, recall: f64) -> PyResult<()> {
        if let Some(metrics) = &self.metrics { metrics.update_recall_estimate(k, recall); }
        Ok(())
    }
}

impl AnnIndex {
    fn inner_search(&self, q: &[f32], q_sq: f32, k: usize) -> PyResult<(Vec<i64>, Vec<f32>)> {
        if q.len() != self.dim {
            return Err(RustAnnError::py_err("Dimension Error", format!("Expected dimension {}, got {}", self.dim, q.len())));
        }
        let (ids, dists) = crate::utils::compute_distances_with_ids(&self.entries, q, q_sq, self.metric.clone(), self.minkowski_p, k);
        Ok((ids, dists))
    }

    fn validate_path(path: &str) -> PyResult<()> {
        if path.contains("..") { return Err(RustAnnError::py_err("InvalidPath", "Path must not contain traversal sequences")); }
        Ok(())
    }
}

impl AnnBackend for AnnIndex {
    fn new(dim: usize, metric: Distance) -> Self { AnnIndex { dim, metric, minkowski_p: None, entries: Vec::new(), metrics: None } }
    fn add_item(&mut self, item: Vec<f32>) {
        let id = self.entries.len() as i64;
        let sq = item.iter().map(|x| x * x).sum::<f32>();
        self.entries.push((id, item, sq));
    }
    fn build(&mut self) {}
    fn search(&self, vector: &[f32], k: usize) -> Vec<usize> { self.inner_search(vector, vector.iter().map(|x| x*x).sum(), k).unwrap().0.into_iter().map(|id| id as usize).collect() }
    fn save(&self, path: &str) { let _ = save_index(self, path); }
    fn load(path: &str) -> Self { load_index(path).unwrap() }
}