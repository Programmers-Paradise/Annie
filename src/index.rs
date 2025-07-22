use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, IntoPyArray};
use ndarray::Array2;
use ndarray::s;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use std::sync::{Arc, Mutex};
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use bit_vec::BitVec;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use crate::backend::AnnBackend;
use crate::storage::{save_index, load_index};
use crate::metrics::Distance;
use crate::errors::RustAnnError;
use crate::monitoring::MetricsCollector;
use crate::filters::Filter;

#[pyclass]
#[derive(Serialize, Deserialize)]
/// A brute-force k-NN index with cached norms, Rayon parallelism,
/// built-in filters, and support for multiple distance metrics.
pub struct AnnIndex {
    pub(crate) dim: usize,
    pub(crate) metric: Distance,
    /// If Some(p), use Minkowski-p distance instead of `metric`.
    pub(crate) minkowski_p: Option<f32>,
    /// Stored entries as (id, vector, squared_norm) tuples.
    pub(crate) entries: Vec<Option<(i64, Vec<f32>, f32)>>,
    /// Tracks deleted entries for compaction
    pub(crate) deleted_count: usize,
    /// Maximum allowed deleted entries before compaction
    pub(crate) max_deleted_ratio: f32,
    /// Optional metrics collector for monitoring
    #[serde(skip)]
    pub(crate) metrics: Option<Arc<MetricsCollector>>,
    #[serde(skip)]
    pub(crate) boolean_filters: Mutex<HashMap<String, BitVec>>,
    #[serde(skip)]
    pub(crate) version: Arc<AtomicU64>,
}

#[derive(PartialEq, Debug)]
struct FloatOrd(f32);

impl Eq for FloatOrd {}

impl Ord for FloatOrd {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or_else(|| {
            if self.0.is_nan() && other.0.is_nan() {
                std::cmp::Ordering::Equal
            } else if self.0.is_nan() {
                std::cmp::Ordering::Greater
            } else if other.0.is_nan() {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        })
    }
}

impl PartialOrd for FloatOrd {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[pymethods]
impl AnnIndex {
    #[new]
    /// Create a new index for unit-variant metrics.
    pub fn new(dim: usize, metric: Distance) -> PyResult<Self> {
        if dim == 0 {
            return Err(RustAnnError::py_err("Invalid Dimension", "Dimension must be > 0"));
        }
        Ok(AnnIndex {
            dim,
            metric,
            minkowski_p: None,
            entries: Vec::new(),
            deleted_count: 0,
            max_deleted_ratio: 0.2, // 20% deleted triggers compaction
            metrics: None,
            boolean_filters: Mutex::new(HashMap::new()),
            version: Arc::new(AtomicU64::new(0)),
        })
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
        Ok(AnnIndex {
            dim,
            metric: Distance::Minkowski(p),
            minkowski_p: Some(p),
            entries: Vec::new(),
            deleted_count: 0,
            max_deleted_ratio: 0.2,
            metrics: None,
            boolean_filters: Mutex::new(HashMap::new()),
            version: Arc::new(AtomicU64::new(0)),
        })
    }

    #[staticmethod]
    /// Create a new index with a named metric.
    pub fn new_with_metric(dim: usize, metric_name: &str) -> PyResult<Self> {
        if dim == 0 {
            return Err(RustAnnError::py_err("Invalid Dimension", "Dimension must be > 0"));
        }
        let metric = Distance::new(metric_name);
        Ok(AnnIndex {
            dim,
            metric,
            minkowski_p: None,
            entries: Vec::new(),
            deleted_count: 0,
            max_deleted_ratio: 0.2,
            metrics: None,
            boolean_filters: Mutex::new(HashMap::new()),
            version: Arc::new(AtomicU64::new(0)),
        })
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

        // Check for duplicate IDs
        let existing_ids: HashSet<i64> = self.entries
            .par_iter()
            .with_min_len(1000)
            .filter_map(|e| e.as_ref().map(|(id, _, _)| *id))
            .collect();
        let duplicates: Vec<_> = ids.iter()
            .filter(|id| existing_ids.contains(id))
            .copied()
            .collect();
        if !duplicates.is_empty() {
            return Err(RustAnnError::py_err("Duplicate IDs", format!("Found duplicate IDs: {:?}", duplicates)));
        }

        self.entries.reserve(n);
        let chunk_size = 1000; // vectors per chunk
        let num_chunks = (n + chunk_size - 1) / chunk_size;
        
        for idx in 0..num_chunks {
            let start = idx * chunk_size;
            let end = (start + chunk_size).min(n);
            let chunk_view = view.slice(s![start..end, ..]);
            let chunk_ids = &ids[start..end];
            
            let new_entries: Vec<Option<(i64, Vec<f32>, f32)>> = chunk_view.outer_iter()
                .zip(chunk_ids)
                .par_bridge()
                .map(|(row, &id)| {
                    let v = row.to_vec();
                    let sq_norm = v.iter().map(|x| x * x).sum::<f32>();
                    Some((id, v, sq_norm))
                })
                .collect();
            self.entries.extend(new_entries);
            
            if let Some(cb) = &progress_callback {
                Python::with_gil(|py| -> PyResult<()> {
                    cb.call1(py, (end, n)).map_err(|e| {
                        RustAnnError::py_err("Callback Error", format!("Progress callback failed: {}", e))
                    })?;
                    Ok(())
                })?;
            }
        }
        
        // Increment version
        self.version.fetch_add(1, AtomicOrdering::Relaxed);
        
        // Invalidate boolean filters since entries changed
        self.boolean_filters.lock().unwrap().clear();
        
        if let Some(metrics) = &self.metrics {
            let metric_name = match &self.metric {
                Distance::Euclidean() => "euclidean".to_string(),
                Distance::Cosine() => "cosine".to_string(),
                Distance::Manhattan() => "manhattan".to_string(),
                Distance::Chebyshev() => "chebyshev".to_string(),
                Distance::Minkowski(p) => format!("minkowski_p{}", p),
                Distance::Hamming() => "hamming".to_string(),
                Distance::Jaccard() => "jaccard".to_string(),
                Distance::Angular() => "angular".to_string(),
                Distance::Canberra() => "canberra".to_string(),
                Distance::Custom(n) => n.clone(),
            };
            metrics.set_index_metadata(self.len(), self.dim, &metric_name);
        }
        Ok(())
    }

    /// Remove entries by ID.
    pub fn remove(&mut self, ids: Vec<i64>) -> PyResult<()> {
        if ids.is_empty() {
            return Ok(());
        }
        
        let to_remove: HashSet<i64> = ids.into_iter().collect();
        let mut removed_count = 0;
        
        for entry in &mut self.entries {
            if let Some((id, _, _)) = entry {
                if to_remove.contains(id) {
                    *entry = None;
                    removed_count += 1;
                }
            }
        }
        
        self.deleted_count += removed_count;
        
        // Always compact if deleted_count is large, or provide a manual compact option for users to reclaim memory immediately.
        if self.should_compact() || self.deleted_count > 100_000 {
            self.compact()?;
        }
        Ok(())
    }

    pub fn update(&mut self, id: i64, vector: Vec<f32>) -> PyResult<()> {
        if vector.len() != self.dim {
            return Err(RustAnnError::py_err(
                "Dimension Error", 
                format!("Expected dimension {}, got {}", self.dim, vector.len())
            ));
        }
        
        let mut found = false;
        let sq_norm = vector.iter().map(|x| x * x).sum::<f32>();
        
        for entry in &mut self.entries {
            if let Some((entry_id, ref mut vec, ref mut norm)) = entry {
                if *entry_id == id {
                    *vec = vector.clone();
                    *norm = sq_norm;
                    found = true;
                    break;
                }
            }
        }
        
        if !found {
            return Err(RustAnnError::py_err(
                "ID Not Found", 
                format!("ID {} not found in index", id)
            ));
        }
        
        // Increment version
        self.version.fetch_add(1, AtomicOrdering::Relaxed);
        
        Ok(())
    }

    /// Compact index by removing deleted entries
    pub fn compact(&mut self) -> PyResult<()> {
        if self.deleted_count == 0 {
            return Ok(());
        }
        
        let _original_len = self.entries.len();
        self.entries.retain(|e| e.is_some());
        self.deleted_count = 0;
        
        // Increment version
        self.version.fetch_add(1, AtomicOrdering::Relaxed);
        
        // Clear filters after compaction
        self.boolean_filters.lock().unwrap().clear();
        
        if let Some(_metrics) = &self.metrics {
        //    (**metrics).record_compaction(original_len, self.entries.len());
        }
        
        Ok(())
    }

    /// Set maximum deleted ratio before auto-compaction
    pub fn set_max_deleted_ratio(&mut self, ratio: f32) -> PyResult<()> {
        if !(0.0..=1.0).contains(&ratio) {
            return Err(RustAnnError::py_err(
                "Invalid Ratio", 
                "Ratio must be between 0.0 and 1.0"
            ));
        }
        self.max_deleted_ratio = ratio;
        Ok(())
    }

    /// Single query search with optional filter.
    pub fn search(
        &self,
        py: Python,
        query: PyReadonlyArray1<f32>,
        k: usize,
        filter: Option<Filter>
    ) -> PyResult<(PyObject, PyObject)> {
        if self.entries.is_empty() || self.len() == 0 {
            return Err(RustAnnError::py_err("EmptyIndex", "Index is empty"));
        }
        let q = query.as_slice()?;
        let q_sq = q.iter().map(|x| x * x).sum::<f32>();
        let start = Instant::now();
        
        let version = self.version.load(AtomicOrdering::Relaxed);

        let (ids, dists) = py.allow_threads(|| {
            self.inner_search(q, q_sq, k, filter.as_ref(), version)
        })?;
        
        if let Some(metrics) = &self.metrics {
            metrics.record_query(start.elapsed());
        }
        Ok((ids.into_pyarray(py).into(), dists.into_pyarray(py).into()))
    }

    /// Batch queries search with optional filter.
    pub fn search_batch(
        &self,
        py: Python,
        data: PyReadonlyArray2<f32>,
        k: usize,
        filter: Option<Filter>
    ) -> PyResult<(PyObject, PyObject)> {
        let arr = data.as_array();
        let n = arr.nrows();
        if arr.ncols() != self.dim {
            return Err(RustAnnError::py_err("Dimension Error", format!("Expected shape (N, {}), got (N, {})", self.dim, arr.ncols())));
        }
        
        let version = self.version.load(AtomicOrdering::Relaxed);
        let results: Result<Vec<_>, RustAnnError> = py.allow_threads(|| {
            let filter_ref = filter.as_ref();
            (0..n).into_par_iter().map(|i| {
                let row = arr.row(i).to_vec();
                let q_sq = row.iter().map(|x| x * x).sum::<f32>();
                self.inner_search(&row, q_sq, k, filter_ref, version)
                    .map_err(|e| RustAnnError::io_err(format!("Parallel search failed: {}", e)))
            }).collect()
        });
        
        let results = results.map_err(|e| e.into_pyerr())?;
        let (all_ids, all_dists): (Vec<_>, Vec<_>) = results.into_iter().unzip();
        let ids_arr = Array2::from_shape_vec((n, k), all_ids.concat())
            .map_err(|e| RustAnnError::py_err("Reshape Error", format!("Reshape ids failed: {}", e)))?;
        let dists_arr = Array2::from_shape_vec((n, k), all_dists.concat())
            .map_err(|e| RustAnnError::py_err("Reshape Error", format!("Reshape dists failed: {}", e)))?;
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
    pub fn len(&self) -> usize { 
        self.entries.iter().filter(|e| e.is_some()).count()
    }
    
    pub fn __len__(&self) -> usize { self.len() }

    pub fn capacity(&self) -> usize {
        self.entries.len()
    }
    
    /// Deleted entry count
    pub fn deleted_count(&self) -> usize {
        self.deleted_count
    }
    
    /// Current version
    pub fn version(&self) -> u64 {
        self.version.load(AtomicOrdering::Relaxed)
    }

    /// Vector dimension.
    pub fn dim(&self) -> usize { self.dim }

    /// String repr.
    pub fn __repr__(&self) -> String {
        let m = if let Some(p) = self.minkowski_p { 
            format!("Minkowski(p={})", p) 
        } else { 
            format!("{:?}", self.metric) 
        };
        format!("AnnIndex(dim={}, metric={}, entries={})", self.dim, m, self.entries.len())
    }

    /// Enable metrics on optional port.
    pub fn enable_metrics(&mut self, port: Option<u16>) -> PyResult<()> {
        let metrics = Arc::new(MetricsCollector::new());
        let metric_name = if let Some(p) = self.minkowski_p { 
            format!("minkowski_p{}", p) 
        } else {
            match &self.metric {
                Distance::Euclidean() => "euclidean".to_string(),
                Distance::Cosine() => "cosine".to_string(),
                Distance::Manhattan() => "manhattan".to_string(),
                Distance::Chebyshev() => "chebyshev".to_string(),
                Distance::Minkowski(p) => format!("minkowski_p{}", p),
                Distance::Hamming() => "hamming".to_string(),
                Distance::Jaccard() => "jaccard".to_string(),
                Distance::Angular() => "angular".to_string(),
                Distance::Canberra() => "canberra".to_string(),
                Distance::Custom(n) => n.clone(),
            }
        };
        metrics.set_index_metadata(self.entries.len(), self.dim, &metric_name);
        if let Some(p) = port {
            use crate::monitoring::MetricsServer;
            let server = MetricsServer::new(Arc::clone(&metrics), p);
            server.start().map_err(|e| 
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to start metrics server: {}", e)))?;
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
        if let Some(metrics) = &self.metrics { 
            metrics.update_recall_estimate(k, recall); 
        }
        Ok(())
    }
    
    /// Update a boolean filter by name
    pub fn update_boolean_filter(&self, name: String, bits: Vec<bool>) -> PyResult<()> {
        let mut bv = BitVec::from_elem(bits.len(), false);
        for (i, &bit) in bits.iter().enumerate() {
            bv.set(i, bit);
        }
        let mut filters = self.boolean_filters.lock()
            .map_err(|_| RustAnnError::py_err("LockError", "Failed to acquire boolean filters lock"))?;
        filters.insert(name, bv);
        Ok(())
    }
    
    /// Get a boolean filter by name
    pub fn get_boolean_filter(&self, name: &str) -> PyResult<Option<Vec<bool>>> {
        let filters = self.boolean_filters.lock()
            .map_err(|_| RustAnnError::py_err("LockError", "Failed to acquire boolean filters lock"))?;
        Ok(filters.get(name).map(|bv| bv.iter().collect()))
    }
}

impl AnnIndex {
    fn should_compact(&self) -> bool {
        let total = self.entries.len();
        total > 0 && (self.deleted_count as f32 / total as f32) > self.max_deleted_ratio
    }

    pub fn get_info(&self) -> HashMap<String, String> {
        let mut info = HashMap::new();
        info.insert("type".to_string(), "brute".to_string());
        info.insert("dim".to_string(), self.dim.to_string());
        
        let metric = if let Some(p) = self.minkowski_p {
            format!("Minkowski(p={})", p)
        } else {
            format!("{:?}", self.metric)
        };
        info.insert("metric".to_string(), metric);
        
        info.insert("size".to_string(), self.len().to_string());
        info.insert("capacity".to_string(), self.capacity().to_string());
        info.insert("deleted_count".to_string(), self.deleted_count.to_string());
        info.insert("max_deleted_ratio".to_string(), self.max_deleted_ratio.to_string());
        info.insert("version".to_string(), self.version().to_string());
        
        // Calculate memory usage
        let entry_size = std::mem::size_of::<Option<(i64, Vec<f32>, f32)>>();
        let entry_overhead = self.entries.capacity() * entry_size;
        let vector_data = self.len() * self.dim * 4; // 4 bytes per f32
        let norms = self.len() * 4; // 4 bytes per norm
        let total_memory = entry_overhead + vector_data + norms;
        info.insert("memory_bytes".to_string(), total_memory.to_string());
        
        info
    }

    /// Validate index integrity
    pub fn validate(&self) -> PyResult<()> {
        let mut seen_ids = HashSet::new();
        let mut errors = Vec::new();

        for (idx, entry) in self.entries.iter().enumerate() {
            if let Some((id, vec, stored_norm)) = entry {
                // Check ID uniqueness
                if !seen_ids.insert(*id) {
                    errors.push(format!("Duplicate ID found: {}", id));
                }

                // Check vector dimension
                if vec.len() != self.dim {
                    errors.push(format!(
                        "Vector {} (index {}) has dimension {}, expected {}",
                        id, idx, vec.len(), self.dim
                    ));
                }

                // Check norm matches vector
                let computed_norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                if (computed_norm - *stored_norm).abs() > 0.001 {
                    errors.push(format!(
                        "Vector {} (index {}) has norm {} but computed {}",
                        id, idx, stored_norm, computed_norm
                    ));
                }
            }
        }

        if !errors.is_empty() {
            return Err(RustAnnError::py_err(
                "ValidationError",
                format!("{} issues found:\n{}", errors.len(), errors.join("\n"))
            ));
        }
        Ok(())
    }

    fn inner_search(
        &self,
        q: &[f32],
        q_sq: f32,
        k: usize,
        filter: Option<&Filter>,
        version: u64,
    ) -> PyResult<(Vec<i64>, Vec<f32>)> {
        if version != self.version.load(AtomicOrdering::Relaxed) {
            return Err(RustAnnError::py_err(
                "ConcurrentModification", 
                "Index modified during search operation"
            ));
        }

        if q.len() != self.dim {
            return Err(RustAnnError::py_err("Dimension Error", format!("Expected dimension {}, got {}", self.dim, q.len())));
        }

        let candidates: Vec<(i64, f32)> = self.entries
            .par_iter()
            .enumerate()
            .filter_map(|(idx, entry_opt)| {
                // skip deleted entries
                 let (id, vec, sq_norm) = entry_opt.as_ref()?;
                // apply user-provided filter
                if let Some(f) = filter {
                    if !f.accepts(*id, idx) {
                        return None;
                    }
                }
                // compute the distance
                let dist = match self.metric {
                    Distance::Euclidean()   => crate::metrics::euclidean_sq(q, vec, q_sq, *sq_norm),
                    Distance::Cosine()      => crate::metrics::angular_distance(q, vec, q_sq, *sq_norm),
                    Distance::Manhattan()   => crate::metrics::manhattan(q, vec),
                    Distance::Chebyshev()   => crate::metrics::chebyshev(q, vec),
                    Distance::Minkowski(p)  => crate::metrics::minkowski(q, vec, p),
                    Distance::Hamming()     => crate::metrics::hamming(q, vec),
                    Distance::Jaccard()     => crate::metrics::jaccard(q, vec),
                    Distance::Angular()     => crate::metrics::angular_distance(q, vec, q_sq, *sq_norm),
                    Distance::Canberra()    => crate::metrics::canberra(q, vec),
                    Distance::Custom(_) => return None, // or error out
                };
                Some((*id, dist))
            })
            .collect();
        
        if candidates.is_empty() {
            return Ok((vec![], vec![]));
        }

        // Use a min-heap to select top k efficiently
        use std::cmp::Ordering;
        
        let k = k.min(candidates.len());
        if k == 0 {
            return Ok((vec![], vec![]));
        }
        
        let mut candidates = candidates;
        let (left, mid, _) = candidates.select_nth_unstable_by(k - 1, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or_else(|| {
                // Handle NaN values consistently
                if a.1.is_nan() && b.1.is_nan() {
                    Ordering::Equal
                } else if a.1.is_nan() {
                    Ordering::Greater
                } else if b.1.is_nan() {
                    Ordering::Less
                } else {
                    Ordering::Equal
                }
            })
        });

        // Collect and sort only the top-k candidates
        let mut top_k = left.to_vec();
        top_k.push(*mid);
        top_k.sort_unstable_by(|a, b| {
            a.1.partial_cmp(&b.1).unwrap_or_else(|| {
                if a.1.is_nan() && b.1.is_nan() {
                    Ordering::Equal
                } else if a.1.is_nan() {
                    Ordering::Greater
                } else if b.1.is_nan() {
                    Ordering::Less
                } else {
                    Ordering::Equal
                }
            })
        });

        // Extract results
        let (ids, dists): (Vec<_>, Vec<_>) = top_k.into_iter().unzip();
        Ok((ids, dists))
    }

    fn validate_path(path: &str) -> PyResult<()> {
        if path.contains("..") { 
            return Err(RustAnnError::py_err("InvalidPath", "Path must not contain traversal sequences")); 
        }
        Ok(())
    }
}

impl AnnBackend for AnnIndex {
    fn new(dim: usize, metric: Distance) -> Self {
        AnnIndex {
            dim,
            metric,
            minkowski_p: None,
            entries: Vec::new(),
            deleted_count: 0,
            max_deleted_ratio: 0.2,
            metrics: None,
            boolean_filters: Mutex::new(HashMap::new()),
            version: Arc::new(AtomicU64::new(0)),
        }
    }
    
    fn add_item(&mut self, item: Vec<f32>) {
        let id = self.entries.len() as i64;
        let sq = item.iter().map(|x| x * x).sum::<f32>();
        self.entries.push(Some((id, item, sq)));
    }
    
    fn build(&mut self) {}
    
    fn search(&self, vector: &[f32], k: usize) -> Vec<usize> { 
        let q_sq = vector.iter().map(|x| x * x).sum();
         self.inner_search(vector, q_sq, k, None, self.version())
            .unwrap_or_default()
            .0
            .into_iter()
            .map(|id| id as usize)
            .collect()
    }
    
    fn save(&self, path: &str) { 
        let _ = save_index(self, path); 
    }
    
    fn load(path: &str) -> Self { 
        load_index(path).unwrap() 
    }
}