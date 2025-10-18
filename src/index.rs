use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use ndarray::Array2;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use std::sync::{Arc, Mutex};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use bit_vec::BitVec;

use crate::backend::AnnBackend;
use crate::storage::{save_index, load_index};
use crate::metrics::Distance;
use crate::errors::RustAnnError;
use crate::filters::Filter;
use crate::monitoring::MetricsCollector;
use crate::path_validation::validate_path_secure;
#[pyclass]
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum MetadataType {
    Int,
    Float,
    String,
    Tags,
    Timestamp,
}

#[pymethods]
impl MetadataType {
    #[new]
    fn new() -> Self {
        MetadataType::String // Default
    }
}

#[pyclass]
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct MetadataField {
    #[pyo3(get, set)]
    pub field_type: MetadataType,
}

#[pymethods]
impl MetadataField {
    #[new]
    fn new(field_type: MetadataType) -> Self {
        MetadataField { field_type }
    }
}

#[pyclass]
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum MetadataValue {
    Int(i64),
    Float(f64),
    String(String),
    Tags(Vec<String>),
    Timestamp(i64), // Unix timestamp
}

#[pymethods]
impl MetadataValue {
    #[new]
    fn new() -> Self {
        MetadataValue::String(String::new()) // Default
    }
}

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
    pub(crate) metrics: Option<Arc<crate::monitoring::MetricsCollector>>,
    #[serde(skip)]
    pub(crate) boolean_filters: Mutex<HashMap<String, bit_vec::BitVec>>,
    #[serde(skip)]
    pub(crate) version: Arc<AtomicU64>,
    /// Metadata schema: field name -> type
    pub(crate) metadata_schema: Option<HashMap<String, MetadataType>>,
    /// Columnar metadata storage: field name -> Vec of values
    pub(crate) metadata_columns: Option<HashMap<String, Vec<MetadataValue>>>,
}

#[pymethods]
impl AnnIndex {
    /// Set metadata schema from Python (dict: str -> MetadataField)
    pub fn py_set_metadata_schema(&mut self, schema: Bound<'_, pyo3::PyAny>) -> PyResult<()> {
        use pyo3::types::PyDict;
        
        let dict = schema.downcast::<PyDict>()?;
        
        let mut schema_map = HashMap::new();
        for (key, value) in dict.iter() {
            let field_name: String = key.extract()?;
            let metadata_field: MetadataField = value.extract()?;
            schema_map.insert(field_name, metadata_field.field_type);
        }
        
        self.metadata_schema = Some(schema_map);
        Ok(())
    }

    /// Add vectors, IDs, and metadata from Python
    pub fn py_add_with_metadata(&mut self, py: Python, data: PyReadonlyArray2<f32>, ids: PyReadonlyArray1<i64>, metadata: Bound<'_, pyo3::PyAny>) -> PyResult<()> {
        use pyo3::types::PyList;
        
        // Parse metadata list of dictionaries
        let metadata_list = metadata.downcast::<PyList>()?;
        
        let mut parsed_metadata = Vec::new();
        for item in metadata_list.iter() {
            let mut meta_dict = HashMap::new();
            let dict = item.downcast::<pyo3::types::PyDict>()?;
            
            for (key, value) in dict.iter() {
                let field_name: String = key.extract()?;
                let metadata_value = self.parse_metadata_value(&value)?;
                meta_dict.insert(field_name, metadata_value);
            }
            parsed_metadata.push(meta_dict);
        }
        
        self.add_with_metadata_internal(py, data, ids, parsed_metadata)
    }
    
    /// Helper method to parse Python values into MetadataValue
    fn parse_metadata_value(&self, value: &Bound<'_, pyo3::PyAny>) -> PyResult<MetadataValue> {
        // Try extracting different types
        if let Ok(int_val) = value.extract::<i64>() {
            return Ok(MetadataValue::Int(int_val));
        }
        if let Ok(float_val) = value.extract::<f64>() {
            return Ok(MetadataValue::Float(float_val));
        }
        if let Ok(string_val) = value.extract::<String>() {
            return Ok(MetadataValue::String(string_val));
        }
        if let Ok(list_val) = value.extract::<Vec<String>>() {
            return Ok(MetadataValue::Tags(list_val));
        }
        
        Err(pyo3::exceptions::PyTypeError::new_err("Unsupported metadata value type"))
    }

    /// Search with metadata-aware filtering from Python
    pub fn py_search_filtered(&self, query: Vec<f32>, k: usize, predicate: &str) -> PyResult<(Vec<i64>, Vec<f32>)> {
        self.search_filtered(query, k, predicate)
    }
    
    /// Search with metadata-aware filtering using a predicate string
    pub fn search_filtered(&self, query: Vec<f32>, k: usize, _predicate: &str) -> PyResult<(Vec<i64>, Vec<f32>)> {
        // Simple stub implementation for now - just return normal search results
        // TODO: Implement full predicate evaluation
        if query.len() != self.dim {
            return Err(RustAnnError::py_err("Dimension Error", format!("Expected dimension {}, got {}", self.dim, query.len())));
        }
        
        // For now, just do a normal search without filtering 
        // (predicate evaluation will be added in next iterations)
        let mut results: Vec<(i64, f32)> = self.entries
            .iter()
            .filter_map(|entry_opt| {
                if let Some((id, vector, _norm)) = entry_opt {
                    // Simple Euclidean distance
                    let dist = query.iter().zip(vector.iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f32>()
                        .sqrt();
                    Some((*id, dist))
                } else {
                    None
                }
            })
            .collect();
            
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let ids = results.iter().take(k).map(|(id, _)| *id).collect();
        let dists = results.iter().take(k).map(|(_, dist)| *dist).collect();
        Ok((ids, dists))
    }

    #[new]
    /// Create a new index for unit-variant metrics.
    pub fn new(dim: usize, metric: Distance) -> PyResult<Self> {
        const MAX_DIM: usize = 4096;
        if dim == 0 {
            return Err(RustAnnError::py_err("Invalid Dimension", "Dimension must be > 0"));
        }
        if dim > MAX_DIM {
            return Err(RustAnnError::py_err("Excessive Dimension", format!("Dimension {} exceeds safe limit {}", dim, MAX_DIM)));
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
            metadata_schema: None,
            metadata_columns: None,
        })
    }

    #[staticmethod]
    /// Create a new index with Minkowski-p distance.
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
            metadata_schema: None,
            metadata_columns: None,
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
            metadata_schema: None,
            metadata_columns: None,
        })
    }

    /// Add vectors and IDs in batch.
    pub fn add(&mut self, _py: Python, data: PyReadonlyArray2<f32>, ids: PyReadonlyArray1<i64>) -> PyResult<()> {
        // Validate IDs: no negative values
        let ids_slice = ids.as_slice()?;
        if ids_slice.iter().any(|&id| id < 0) {
            return Err(RustAnnError::py_err("Invalid ID", "IDs must be non-negative"));
        }
        self.add_batch_internal(data, ids, None)
    }

    /// Add vectors, IDs, and metadata in batch.
    pub fn add_with_metadata_internal(&mut self, _py: Python, data: PyReadonlyArray2<f32>, ids: PyReadonlyArray1<i64>, metadata: Vec<HashMap<String, MetadataValue>>) -> PyResult<()> {
        let ids_slice = ids.as_slice()?;
        if ids_slice.iter().any(|&id| id < 0) {
            return Err(RustAnnError::py_err("Invalid ID", "IDs must be non-negative"));
        }
        let n = ids_slice.len();
        if metadata.len() != n {
            return Err(RustAnnError::py_err("Metadata Mismatch", "Metadata length must match number of vectors"));
        }
        // Validate schema
        if let Some(schema) = &self.metadata_schema {
            for meta in &metadata {
                for (field, value) in meta {
                    if let Some(expected_type) = schema.get(field) {
                        let valid = match (expected_type, value) {
                            (MetadataType::Int, MetadataValue::Int(_)) => true,
                            (MetadataType::Float, MetadataValue::Float(_)) => true,
                            (MetadataType::String, MetadataValue::String(_)) => true,
                            (MetadataType::Tags, MetadataValue::Tags(_)) => true,
                            (MetadataType::Timestamp, MetadataValue::Timestamp(_)) => true,
                            _ => false,
                        };
                        if !valid {
                            return Err(RustAnnError::py_err("Metadata Type Error", format!("Field '{}' type mismatch", field)));
                        }
                    } else {
                        return Err(RustAnnError::py_err("Unknown Metadata Field", format!("Field '{}' not in schema", field)));
                    }
                }
            }
        }
        // Store metadata column-wise
        if self.metadata_columns.is_none() {
            self.metadata_columns = Some(HashMap::new());
        }
        let columns = self.metadata_columns.as_mut().unwrap();
        if let Some(schema) = &self.metadata_schema {
            for (field, field_type) in schema {
                let mut col: Vec<MetadataValue> = Vec::with_capacity(n);
                for meta in &metadata {
                    if let Some(val) = meta.get(field) {
                        col.push(val.clone());
                    } else {
                        // Default value for missing field
                        col.push(match field_type {
                            MetadataType::Int => MetadataValue::Int(0),
                            MetadataType::Float => MetadataValue::Float(0.0),
                            MetadataType::String => MetadataValue::String(String::new()),
                            MetadataType::Tags => MetadataValue::Tags(Vec::new()),
                            MetadataType::Timestamp => MetadataValue::Timestamp(0),
                        });
                    }
                }
                columns.entry(field.clone()).or_insert(col);
            }
        }
        // Add vectors and IDs as usual
        self.add_batch_internal(data, ids, None)
    }

    /// Add vectors and IDs in batch with progress reporting.
    /// The callback should be a callable that takes two integers: (current, total)
    pub fn add_batch_with_progress(
        &mut self,
    _py: Python,
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
        _progress_callback: Option<PyObject>
    ) -> PyResult<()> {
        let view = data.as_array();
        let ids = ids.as_slice()?;
        let n = view.nrows();
        const MAX_ROWS: usize = 1_000_000;
        if n != ids.len() {
            return Err(RustAnnError::py_err("Input Mismatch", "`data` and `ids` must have same length"));
        }
        if view.ncols() != self.dim {
            return Err(RustAnnError::py_err("Dimension Error", format!("Expected dimension {}, got {}", self.dim, view.ncols())));
        }
        if n > MAX_ROWS {
            return Err(RustAnnError::py_err("Excessive Allocation", format!("Attempted to add {} vectors, limit is {}", n, MAX_ROWS)));
        }
        let active_entries = self.entries.iter().filter(|e| e.is_some()).count();
        if active_entries + n >= MAX_ROWS {
            return Err(RustAnnError::py_err("Excessive Allocation", format!("Total active entries would reach or exceed safe limit {}", MAX_ROWS)));
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

        // Reserve and insert entries with computed squared norms
        self.entries.reserve(n);
        for (row_idx, id) in ids.iter().enumerate() {
            let row = view.row(row_idx);
            let vec: Vec<f32> = row.to_vec();
            let sq_norm: f32 = vec.iter().map(|x| x * x).sum();
            self.entries.push(Some((*id, vec, sq_norm)));
        }

        // Bump version to signal mutation
        self.version.fetch_add(1, AtomicOrdering::Relaxed);

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
        if q.len() != self.dim {
            return Err(RustAnnError::py_err(
                "Dimension Error",
                format!("Expected dimension {}, got {}", self.dim, q.len()),
            ));
        }
        let q_sq = q.iter().map(|x| x * x).sum::<f32>();
        let version = self.version.load(AtomicOrdering::Relaxed);
        let (ids, dists) = self.inner_search(q, q_sq, k, filter.as_ref(), version)?;

        let ids_array = numpy::PyArray1::from_vec(py, ids).into_any();
        let dists_array = numpy::PyArray1::from_vec(py, dists).into_any();

        Ok((ids_array.unbind(), dists_array.unbind()))
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
        
        let _version = self.version.load(AtomicOrdering::Relaxed);
        let results: Result<Vec<_>, RustAnnError> = py.allow_threads(|| {
            let filter_ref = filter.as_ref();
            (0..n).into_par_iter().map(|i| {
                let row = arr.row(i).to_vec();
                // Simple search for each row (replacing inner_search call)
                let mut results: Vec<(i64, f32)> = self.entries
                    .iter()
                    .filter_map(|entry_opt| {
                        if let Some((id, vector, _norm)) = entry_opt {
                            if let Some(f) = filter_ref {
                                if !f.accepts(*id, 0) { // Use 0 as index for now
                                    return None;
                                }
                            }
                            // Simple Euclidean distance
                            let dist = row.iter().zip(vector.iter())
                                .map(|(a, b)| (a - b) * (a - b))
                                .sum::<f32>()
                                .sqrt();
                            Some((*id, dist))
                        } else {
                            None
                        }
                    })
                    .collect();
                    
                results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                let ids: Vec<i64> = results.iter().take(k).map(|(id, _)| *id).collect();
                let dists: Vec<f32> = results.iter().take(k).map(|(_, dist)| *dist).collect();
                Ok((ids, dists))
            }).collect()
        });
        
        let results = results.map_err(|e| e.into_pyerr())?;
        let (all_ids, all_dists): (Vec<_>, Vec<_>) = results.into_iter().unzip();
        let ids_arr = Array2::from_shape_vec((n, k), all_ids.concat())
            .map_err(|e| RustAnnError::py_err("Reshape Error", format!("Reshape ids failed: {}", e)))?;
        let dists_arr = Array2::from_shape_vec((n, k), all_dists.concat())
            .map_err(|e| RustAnnError::py_err("Reshape Error", format!("Reshape dists failed: {}", e)))?;
    Ok((ids_arr.to_pyarray(py).into(), dists_arr.to_pyarray(py).into()))
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

    /// Remove entries by IDs (not yet implemented)
    pub fn remove(&mut self, _ids: Vec<i64>) -> PyResult<()> {
        Err(RustAnnError::py_err("NotImplemented", "Remove operation not yet implemented"))
    }

    /// Update an entry by ID (not yet implemented)
    pub fn update(&mut self, _id: i64, _vector: Vec<f32>) -> PyResult<()> {
        Err(RustAnnError::py_err("NotImplemented", "Update operation not yet implemented"))
    }

    /// Compact the index by removing deleted entries (not yet implemented)
    pub fn compact(&mut self) -> PyResult<()> {
        Err(RustAnnError::py_err("NotImplemented", "Compact operation not yet implemented"))
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

}

// Private methods implementation (not exposed to Python)
impl AnnIndex {
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
            safe_partial_cmp(&a.1, &b.1)
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
    let ids: Vec<i64> = top_k.iter().map(|(id, _)| *id).collect();
    let dists: Vec<f32> = top_k.iter().map(|(_, dist)| *dist).collect();
    Ok((ids, dists))
    }

    /// Secure path validation using canonicalization and allowlist
    /// 
    /// Replaces the vulnerable simple string check with robust path validation
    /// that prevents directory traversal attacks through multiple bypass techniques.
    fn validate_path(path: &str) -> PyResult<()> {
        // Use the secure path validation module
        validate_path_secure(path).map(|_| ())
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
            metadata_schema: None,
            metadata_columns: None,
        }
    }
    
    fn add_item(&mut self, item: Vec<f32>) {
        let id = self.entries.len() as i64;
        let sq = item.iter().map(|x| x * x).sum::<f32>();
        self.entries.push(Some((id, item, sq)));
    }
    
    fn build(&mut self) {}
    
    fn search(&self, vector: &[f32], k: usize) -> Vec<usize> { 
        // Simple search implementation without inner_search
        let mut results: Vec<(usize, f32)> = self.entries
            .iter()
            .enumerate()
            .filter_map(|(idx, entry_opt)| {
                if let Some((_id, vec, _norm)) = entry_opt {
                    // Simple Euclidean distance
                    let dist = vector.iter().zip(vec.iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f32>()
                        .sqrt();
                    Some((idx, dist))
                } else {
                    None
                }
            })
            .collect();
            
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.iter().take(k).map(|(idx, _)| *idx).collect()
    }
    
    fn save(&self, path: &str) { 
        let _ = save_index(self, path); 
    }
    

    fn load(path: &str) -> Self { 
        load_index(path).unwrap() 
    }
}


fn safe_partial_cmp(a: &f32, b: &f32) -> std::cmp::Ordering {
    a.partial_cmp(b).unwrap_or_else(|| {
        if a.is_nan() && b.is_nan() {
            std::cmp::Ordering::Equal
        } else if a.is_nan() {
            std::cmp::Ordering::Greater
        } else if b.is_nan() {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Equal
        }
    })
}
