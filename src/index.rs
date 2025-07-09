use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, IntoPyArray};
use ndarray::Array2;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};

use crate::backend::AnnBackend;  //new added
use crate::storage::{save_index, load_index};
use crate::metrics::Distance;
use crate::errors::RustAnnError;

/// A brute-force k-NN index with cached norms, Rayon parallelism,
/// and support for L1, L2, Cosine, Chebyshev, and Minkowski-p distances.
/// 
/// /// Args:
///     dim (int): Vector dimension
///     metric (Distance): Distance metric to use
///
/// Example:
///     index = AnnIndex(128, Distance.EUCLIDEAN)
#[pyclass]
#[derive(Serialize, Deserialize)]
pub struct AnnIndex {
    dim: usize,
    metric: Distance,
    /// If Some(p), use Minkowski-p distance instead of `metric`.
    minkowski_p: Option<f32>,
    /// Stored entries as (id, vector, squared_norm) tuples.
    entries: Vec<(i64, Vec<f32>, f32)>,
}

#[pymethods]
impl AnnIndex {
    /// Create a new index for unit-variant metrics (Euclidean, Cosine, Manhattan, Chebyshev).
    /// 
    /// /// Args:
    ///     dim (int): Vector dimension
    ///     metric (Distance): Distance metric
    #[new]
    pub fn new(dim: usize, metric: Distance) -> PyResult<Self> {
        if dim == 0 {
            return Err(RustAnnError::py_err("Invalid Dimension","Dimension must be > 0"));
        }
        Ok(AnnIndex {
            dim,
            metric,
            minkowski_p: None,
            entries: Vec::new(),
        })
    }

    /// Create a new index using Minkowski-p distance (p > 0).
    /// 
    /// Args:
    ///     dim (int): Vector dimension
    ///     p (float): Minkowski exponent (p > 0)
    #[staticmethod]
    pub fn new_minkowski(dim: usize, p: f32) -> PyResult<Self> {
        if dim == 0 {
            return Err(RustAnnError::py_err("Invalid Dimension","Dimension must be > 0"));
        }
        if p <= 0.0 {
            return Err(RustAnnError::py_err("Minkowski Error","`p` must be > 0 for Minkowski distance"));
        }
        Ok(AnnIndex {
            dim,
            metric: Distance::Euclidean, // placeholder
            minkowski_p: Some(p),
            entries: Vec::new(),
        })
    }

    /// Add a batch of vectors (shape: N×dim) with integer IDs.
    /// 
    /// Args:
    ///     data (ndarray): N x dim array of vectors
    ///     ids (ndarray): N-dimensional array of IDs
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
        Ok(())
    }

    /// Remove entries whose IDs appear in `ids`.
    pub fn remove(&mut self, ids: Vec<i64>) -> PyResult<()> {
        if !ids.is_empty() {
            let to_rm: std::collections::HashSet<i64> = ids.into_iter().collect();
            self.entries.retain(|(id, _, _)| !to_rm.contains(id));
        }
        Ok(())
    }

    /// Search the k nearest neighbors for a single query vector.
    /// 
    /// Args:
    ///     query (ndarray): Query vector (dim-dimensional)
    ///     k (int): Number of neighbors to return
    ///
    /// Returns:
    ///     Tuple[ndarray, ndarray]: (neighbor IDs, distances)
    pub fn search(
        &self,
        py: Python,
        query: PyReadonlyArray1<f32>,
        k: usize,
    ) -> PyResult<(PyObject, PyObject)> {
        let q = query.as_slice()?;
        let q_sq = q.iter().map(|x| x * x).sum::<f32>();

        // Release the GIL for the heavy compute:
        let result: PyResult<(Vec<i64>, Vec<f32>)> = py.allow_threads(|| {
            self.inner_search(q, q_sq, k)
        });
        let (ids, dists) = result?;

        Ok((
            ids.into_pyarray(py).into(),
            dists.into_pyarray(py).into(),
        ))
    }

    /// Batch-search k nearest neighbors for each row in an (N×dim) array.
    /// 
    /// Args:
    ///     query (ndarray): Query vector
    ///     k (int): Number of neighbors
    ///     filter_fn (Callable[[int], bool]): Filter function
    ///
    /// Returns:
    ///     Tuple[ndarray, ndarray]: Filtered (neighbor IDs, distances)
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
    
    /// Save index to `<path>.bin`.
    pub fn save(&self, path: &str) -> PyResult<()> {
        let full = format!("{}.bin", path);
        save_index(self, &full).map_err(|e| e.into_pyerr())
    }

    /// Load index from `<path>.bin`.
    #[staticmethod]
    pub fn load(path: &str) -> PyResult<Self> {
        let full = format!("{}.bin", path);
        load_index(&full).map_err(|e| e.into_pyerr())
    }

    /// Get the number of entries in the index.
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    /// Get the dimension of vectors in the index.
    pub fn dim(&self) -> usize {
        if self.entries.is_empty() {
            panic!("Cannot get dimension of empty index");
        }
        self.dim
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
            self.metric,
            self.minkowski_p,
            k,
        );

    Ok((ids, dists))
    }
}
impl AnnBackend for AnnIndex {
    fn new(dim: usize, metric: Distance) -> Self {
        AnnIndex {
            dim,
            metric,
            minkowski_p: None,
            entries: Vec::new(),
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
            self.metric,
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
