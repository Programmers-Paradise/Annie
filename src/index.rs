use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, IntoPyArray};
use ndarray::Array2;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use std::time::Instant;

use crate::backend::AnnBackend;  //new added
use crate::storage::{save_index, load_index};
use crate::metrics::Distance;
use crate::errors::RustAnnError;
use crate::monitoring::MetricsCollector;

#[pyclass]
#[derive(Serialize, Deserialize)]
/// A brute-force k-NN index with cached norms, Rayon parallelism,
/// and support for L1, L2, Cosine, Chebyshev, and Minkowski-p distances.
/// 
/// This class provides efficient nearest neighbor search using various distance metrics
/// with parallel processing capabilities for batch operations.
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
    /// Create a new index for unit-variant metrics (Euclidean, Cosine, Manhattan, Chebyshev).
    /// 
    /// Args:
    ///     dim (int): Vector dimension. Must be greater than 0.
    ///     metric (Distance): Distance metric to use for similarity computation.
    ///                       Options: Distance.EUCLIDEAN, Distance.COSINE, 
    ///                       Distance.MANHATTAN, Distance.CHEBYSHEV.
    ///
    /// Returns:
    ///     AnnIndex: A new empty index instance.
    ///
    /// Raises:
    ///     RustAnnError: If dimension is 0 or invalid.
    ///
    /// Example:
    ///     >>> from annindex import AnnIndex, Distance
    ///     >>> index = AnnIndex(128, Distance.EUCLIDEAN)
    ///     >>> index = AnnIndex(256, Distance.COSINE)
    pub fn new(dim: usize, metric: Distance) -> PyResult<Self> {
        if dim == 0 {
            return Err(RustAnnError::py_err("Invalid Dimension","Dimension must be > 0"));
        }
        Ok(AnnIndex {
            dim,
            metric,
            minkowski_p: None,
            entries: Vec::new(),
            metrics: None,
        })
    }

    #[staticmethod]
    /// Create a new index using Minkowski-p distance (p > 0).
    /// 
    /// Args:
    ///     dim (int): Vector dimension. Must be greater than 0.
    ///     p (float): Minkowski exponent. Must be greater than 0.
    ///               When p=1, equivalent to Manhattan distance.
    ///               When p=2, equivalent to Euclidean distance.
    ///
    /// Returns:
    ///     AnnIndex: A new empty index instance configured for Minkowski-p distance.
    ///
    /// Raises:
    ///     RustAnnError: If dimension is 0 or p <= 0.
    ///
    /// Example:
    ///     >>> index = AnnIndex.new_minkowski(128, 1.5)
    ///     >>> index = AnnIndex.new_minkowski(64, 3.0)
    pub fn new_minkowski(dim: usize, p: f32) -> PyResult<Self> {
        if dim == 0 {
            return Err(RustAnnError::py_err("Invalid Dimension","Dimension must be > 0"));
        }
        if p <= 0.0 {
            return Err(RustAnnError::py_err("Minkowski Error","`p` must be > 0 for Minkowski distance"));
        }
        Ok(AnnIndex {
            dim,
            metric: Distance::Euclidean(), // placeholder
            minkowski_p: Some(p),
            entries: Vec::new(),
            metrics: None,
        })
    }

    #[staticmethod]
    /// Create a new index using a custom distance metric by name.
    /// 
    /// Args:
    ///     dim (int): Vector dimension. Must be greater than 0.
    ///     metric_name (str): Name of the distance metric to use.
    ///                       Can be built-in ("euclidean", "cosine", "manhattan", "chebyshev")
    ///                       or a custom metric registered via register_metric().
    ///
    /// Returns:
    ///     AnnIndex: A new empty index instance.
    ///
    /// Raises:
    ///     RustAnnError: If dimension is 0 or invalid.
    ///
    /// Example:
    ///     >>> index = AnnIndex.new_with_metric(128, "euclidean")
    ///     >>> index = AnnIndex.new_with_metric(64, "my_custom_metric")
    pub fn new_with_metric(dim: usize, metric_name: &str) -> PyResult<Self> {
        if dim == 0 {
            return Err(RustAnnError::py_err("Invalid Dimension","Dimension must be > 0"));
        }
        
        let metric = Distance::new(metric_name);
        
        Ok(AnnIndex {
            dim,
            metric,
            minkowski_p: None,
            entries: Vec::new(),
            metrics: None,
        })
    }

    /// Add a batch of vectors with their corresponding IDs to the index.
    /// 
    /// Args:
    ///     data (numpy.ndarray): N x dim array of vectors to add to the index.
    ///                          Each row represents a vector.
    ///     ids (numpy.ndarray): N-dimensional array of integer IDs corresponding
    ///                         to each vector in data.
    ///
    /// Raises:
    ///     RustAnnError: If data and ids have different lengths, or if any vector
    ///                   has incorrect dimension.
    ///
    /// Example:
    ///     >>> import numpy as np
    ///     >>> data = np.random.rand(100, 128).astype(np.float32)
    ///     >>> ids = np.arange(100, dtype=np.int64)
    ///     >>> index.add(data, ids)
    pub fn add(
        &mut self,
        _py: Python,
        data: PyReadonlyArray2<f32>,
        ids: PyReadonlyArray1<i64>,
    ) -> PyResult<()> {
        let view = data.as_array();
        let ids = ids.as_slice()?;
        if view.nrows() != ids.len() {
            return Err(RustAnnError::py_err("Input Mismatch","`data` and `ids` must have same length"));
        }
        for (row, &id) in view.outer_iter().zip(ids) {
            let v = row.to_vec();
            if v.len() != self.dim {
                return Err(RustAnnError::py_err(
                    "Dimension Error",
                    format!("Expected dimension {}, got {}", self.dim, v.len()))
                );
            }
            let sq_norm = v.iter().map(|x| x * x).sum::<f32>();
            self.entries.push((id, v, sq_norm));
        }
        
        // Update metrics if enabled
        if let Some(metrics) = &self.metrics {
            let metric_name = if let Some(p) = self.minkowski_p {
                format!("minkowski_p{}", p)
            } else {
                match &self.metric {
                    Distance::Euclidean() => "euclidean".to_string(),
                    Distance::Cosine() => "cosine".to_string(),
                    Distance::Manhattan() => "manhattan".to_string(),
                    Distance::Chebyshev() => "chebyshev".to_string(),
                    Distance::Custom(name) => name.clone(),
                }
            };
            metrics.set_index_metadata(self.entries.len(), self.dim, &metric_name);
        }
        
        Ok(())
    }

    /// Remove entries from the index by their IDs.
    /// 
    /// Args:
    ///     ids (List[int]): List of IDs to remove from the index.
    ///                     IDs that don't exist in the index are ignored.
    ///
    /// Example:
    ///     >>> index.remove([1, 5, 10])  # Remove vectors with IDs 1, 5, and 10
    ///     >>> index.remove([])  # No-op for empty list
    pub fn remove(&mut self, ids: Vec<i64>) -> PyResult<()> {
        if !ids.is_empty() {
            let to_rm: std::collections::HashSet<i64> = ids.into_iter().collect();
            self.entries.retain(|(id, _, _)| !to_rm.contains(id));
        }
        Ok(())
    }

    /// Search for the k nearest neighbors of a query vector.
    /// 
    /// Args:
    ///     query (numpy.ndarray): Query vector with dimension matching the index.
    ///                           Should be a 1D array of float32 values.
    ///     k (int): Number of nearest neighbors to return. Must be positive.
    ///
    /// Returns:
    ///     Tuple[numpy.ndarray, numpy.ndarray]: A tuple containing:
    ///         - neighbor_ids: Array of k nearest neighbor IDs (int64)
    ///         - distances: Array of k corresponding distances (float32)
    ///
    /// Raises:
    ///     RustAnnError: If query dimension doesn't match index dimension.
    ///
    /// Example:
    ///     >>> import numpy as np
    ///     >>> query = np.random.rand(128).astype(np.float32)
    ///     >>> neighbor_ids, distances = index.search(query, 10)
    ///     >>> print(f"Found {len(neighbor_ids)} neighbors")
    pub fn search(
        &self,
        py: Python,
        query: PyReadonlyArray1<f32>,
        k: usize,
    ) -> PyResult<(PyObject, PyObject)> {

        if self.entries.is_empty() {
            return Err(RustAnnError::py_err("EmptyIndex", "Index is empty"));
        }

        let q = query.as_slice()?;
        let q_sq = q.iter().map(|x| x * x).sum::<f32>();

        // Record query start time for metrics
        let start_time = Instant::now();

        // Release the GIL for the heavy compute:
        let result: PyResult<(Vec<i64>, Vec<f32>)> = py.allow_threads(|| {
            self.inner_search(q, q_sq, k)
        });
        let (ids, dists) = result?;

        // Record query latency
        if let Some(metrics) = &self.metrics {
            metrics.record_query(start_time.elapsed());
        }

        Ok((
            ids.into_pyarray(py).into(),
            dists.into_pyarray(py).into(),
        ))
    }

    /// Batch search for k nearest neighbors for multiple query vectors.
    /// 
    /// Performs parallel search across multiple query vectors for improved performance
    /// on large batches.
    /// 
    /// Args:
    ///     data (numpy.ndarray): N x dim array of query vectors. Each row is a query.
    ///     k (int): Number of nearest neighbors to return for each query.
    ///
    /// Returns:
    ///     Tuple[numpy.ndarray, numpy.ndarray]: A tuple containing:
    ///         - neighbor_ids: N x k array of neighbor IDs for each query (int64)
    ///         - distances: N x k array of distances for each query (float32)
    ///
    /// Raises:
    ///     RustAnnError: If query dimensions don't match index dimension, or if
    ///                   parallel processing fails.
    ///
    /// Example:
    ///     >>> import numpy as np
    ///     >>> queries = np.random.rand(50, 128).astype(np.float32)
    ///     >>> neighbor_ids, distances = index.search_batch(queries, 5)
    ///     >>> print(f"Shape: {neighbor_ids.shape}")  # (50, 5)
    pub fn search_batch(
        &self,
        py: Python,
        data: PyReadonlyArray2<f32>,
        k: usize,
    ) -> PyResult<(PyObject, PyObject)> {
        let arr = data.as_array();
        let n = arr.nrows();
        
        if arr.ncols() != self.dim {
            return Err(RustAnnError::py_err(
                "Dimension Error",
                format!("Expected query shape (N, {}), got (N, {})", self.dim, arr.ncols())));
        }
        
        // Use allow_threads with parallel iterator and propagate errors as PyResult
        let results: Result<Vec<_>, RustAnnError> = py.allow_threads(|| {
            (0..n)
            .into_par_iter()
            .map(|i| {
                let row = arr.row(i);
                let q: Vec<f32> = row.to_vec();
                let q_sq = q.iter().map(|x| x * x).sum::<f32>();
                self.inner_search(&q, q_sq, k)
                   .map_err(|e| RustAnnError::io_err(format!("Parallel search failed: {}", e)))
            })
            .collect()
        });
        
        // Convert RustAnnError into PyErr before returning
        let results = results.map_err(|e| e.into_pyerr())?;
        
        let (all_ids, all_dists): (Vec<_>, Vec<_>) = results.into_iter().unzip();
        
        let ids_arr: Array2<i64> = Array2::from_shape_vec((n, k), all_ids.concat())
            .map_err(|e| RustAnnError::py_err("Reshape Error", format!("Reshape ids failed: {}", e)))?;
        let dists_arr: Array2<f32> = Array2::from_shape_vec((n, k), all_dists.concat())
            .map_err(|e| RustAnnError::py_err("Reshape Error", format!("Reshape dists failed: {}", e)))?;
        
        Ok((
            ids_arr.into_pyarray(py).into(),
            dists_arr.into_pyarray(py).into(),
        ))
    }
    
    /// Save the index to a binary file.
    /// 
    /// Serializes the entire index (including vectors, IDs, and configuration)
    /// to a binary file with '.bin' extension automatically appended.
    /// 
    /// Args:
    ///     path (str): Base path for the saved file. The '.bin' extension will be
    ///                automatically appended.
    ///
    /// Raises:
    ///     RustAnnError: If the file cannot be written or serialization fails.
    ///
    /// Example:
    ///     >>> index.save("my_index")  # Saves to "my_index.bin"
    ///     >>> index.save("/path/to/index")  # Saves to "/path/to/index.bin"
    pub fn save(&self, path: &str) -> PyResult<()> {
        Self::validate_path(path)?;
        let full = format!("{}.bin", path);
        save_index(self, &full).map_err(|e| e.into_pyerr())
    }

    #[staticmethod]
    /// Load an index from a binary file.
    /// 
    /// Deserializes a previously saved index from a binary file with '.bin' extension
    /// automatically appended.
    /// 
    /// Args:
    ///     path (str): Base path of the saved file. The '.bin' extension will be
    ///                automatically appended.
    ///
    /// Returns:
    ///     AnnIndex: The loaded index instance with all vectors and configuration.
    ///
    /// Raises:
    ///     RustAnnError: If the file cannot be read or deserialization fails.
    ///
    /// Example:
    ///     >>> index = AnnIndex.load("my_index")  # Loads from "my_index.bin"
    ///     >>> index = AnnIndex.load("/path/to/index")  # Loads from "/path/to/index.bin"
    pub fn load(path: &str) -> PyResult<Self> {
        Self::validate_path(path)?;
        let full = format!("{}.bin", path);
        load_index(&full).map_err(|e| e.into_pyerr())
    }

    /// Get the number of vectors currently stored in the index.
    /// 
    /// Returns:
    ///     int: The number of vectors in the index.
    ///
    /// Example:
    ///     >>> len(index)  # This calls __len__ internally
    ///     1000
    ///     >>> index.len()  # Direct method call
    ///     1000
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Get the dimension of vectors stored in the index.
    /// 
    /// Returns:
    ///     int: The vector dimension.
    ///
    /// Raises:
    ///     RuntimeError: If the index is empty (no vectors added yet).
    ///
    /// Example:
    ///     >>> index.dim()
    ///     128
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the number of vectors in the index (implements len()).
    /// 
    /// Returns:
    ///     int: The number of vectors in the index.
    ///
    /// Example:
    ///     >>> len(index)
    ///     1000
    pub fn __len__(&self) -> usize {
        self.entries.len()
    }

    /// Get a string representation of the index.
    /// 
    /// Returns:
    ///     str: A descriptive string showing index statistics.
    ///
    /// Example:
    ///     >>> print(index)
    ///     AnnIndex(dim=128, metric=Euclidean, entries=1000)
    pub fn __repr__(&self) -> String {
        let metric_str = if let Some(p) = self.minkowski_p {
            format!("Minkowski(p={})", p)
        } else {
            format!("{:?}", self.metric)
        };
        format!("AnnIndex(dim={}, metric={}, entries={})", 
                self.dim, metric_str, self.entries.len())
    }

    /// Enable metrics collection for this index.
    /// 
    /// Args:
    ///     port (int): Port number to serve metrics on (default: 8000).
    ///
    /// Example:
    ///     >>> index.enable_metrics(8080)
    ///     >>> # Metrics will be available at http://localhost:8080/metrics
    pub fn enable_metrics(&mut self, port: Option<u16>) -> PyResult<()> {
        let metrics = Arc::new(MetricsCollector::new());
        
        // Update initial metadata
        let metric_name = if let Some(p) = self.minkowski_p {
            format!("minkowski_p{}", p)
        } else {
            match &self.metric {
                Distance::Euclidean() => "euclidean".to_string(),
                Distance::Cosine() => "cosine".to_string(),
                Distance::Manhattan() => "manhattan".to_string(),
                Distance::Chebyshev() => "chebyshev".to_string(),
                Distance::Custom(name) => name.clone(),
            }
        };
        
        metrics.set_index_metadata(self.entries.len(), self.dim, &metric_name);
        
        // Start metrics server if port is specified
        if let Some(port) = port {
            use crate::monitoring::MetricsServer;
            let server = MetricsServer::new(Arc::clone(&metrics), port);
            server.start().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("Failed to start metrics server: {}", e)
                )
            })?;
        }
        
        self.metrics = Some(metrics);
        Ok(())
    }

    /// Get current metrics as a Python dictionary.
    /// 
    /// Returns:
    ///     dict: Dictionary containing current metrics.
    ///
    /// Example:
    ///     >>> metrics = index.get_metrics()
    ///     >>> print(f"Query count: {metrics['query_count']}")
    pub fn get_metrics(&self) -> PyResult<Option<PyObject>> {
        if let Some(metrics) = &self.metrics {
            let snapshot = metrics.get_metrics_snapshot();
            Python::with_gil(|py| {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("query_count", snapshot.query_count)?;
                dict.set_item("avg_query_latency_us", snapshot.avg_query_latency_us)?;
                dict.set_item("index_size", snapshot.index_size)?;
                dict.set_item("dimensions", snapshot.dimensions)?;
                dict.set_item("uptime_seconds", snapshot.uptime_seconds)?;
                dict.set_item("distance_metric", snapshot.distance_metric)?;
                
                // Convert recall estimates to Python dict
                let recall_dict = pyo3::types::PyDict::new(py);
                // In the simplified version, we don't store recall estimates
                dict.set_item("recall_estimates", recall_dict)?;
                
                Ok(Some(dict.into()))
            })
        } else {
            Ok(None)
        }
    }

    /// Update recall estimate for a specific k value.
    /// 
    /// Args:
    ///     k (int): The k value for which to update recall.
    ///     recall (float): The estimated recall value (0.0 to 1.0).
    ///
    /// Example:
    ///     >>> index.update_recall_estimate(10, 0.95)
    pub fn update_recall_estimate(&self, k: usize, recall: f64) -> PyResult<()> {
        if let Some(metrics) = &self.metrics {
            metrics.update_recall_estimate(k, recall);
        }
        Ok(())
    }
}

impl AnnIndex {
    /// Core search logic covering L2, Cosine, L1 (Manhattan), L∞ (Chebyshev), and Lₚ.
    fn inner_search(&self, q: &[f32], q_sq: f32, k: usize) -> PyResult<(Vec<i64>, Vec<f32>)> {
        if q.len() != self.dim {
            return Err(RustAnnError::py_err("Dimension Error",format!(
                "Expected dimension {}, got {}", self.dim, q.len()
            )));
        }

        let (ids, dists) = crate::utils::compute_distances_with_ids(
            &self.entries,
            q,
            q_sq,
            self.metric.clone(),
            self.minkowski_p,
            k,
        );

    Ok((ids, dists))
    }

    fn validate_path(path: &str) -> PyResult<()> {
        if path.contains("..") {
            return Err(RustAnnError::py_err(
                "InvalidPath",
                "Path must not contain traversal sequences"
            ));
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
            metrics: None,
        }
    }

    fn add_item(&mut self, item: Vec<f32>) {
        let id = self.entries.len() as i64;
        let sq_norm = item.iter().map(|x| x * x).sum::<f32>();
        self.entries.push((id, item, sq_norm));
    }

    fn build(&mut self) {
        // No-op for brute-force index
    }

    fn search(&self, vector: &[f32], k: usize) -> Vec<usize> {
        let query_sq = vector.iter().map(|x| x * x).sum::<f32>();

        let (ids, _) = crate::utils::compute_distances_with_ids(
            &self.entries,
            vector,
            query_sq,
            self.metric.clone(),
            self.minkowski_p,
            k,
        );

        ids.into_iter().filter_map(|id| if id >= 0 { Some(id as usize) } else { None }).collect()
    }

    fn save(&self, path: &str) {
        let _ = save_index(self, path);
    }

    fn load(path: &str) -> Self {
        load_index(path).unwrap()
    }
}
