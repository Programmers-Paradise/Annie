// src/index.rs

use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, IntoPyArray};
use ndarray::Array2;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};

use crate::storage::{save_index, load_index};
use crate::metrics::Distance;
use crate::errors::RustAnnError;

/// A brute-force k-NN index with cached norms, Rayon parallelism,
/// and support for L1, L2, Cosine, Chebyshev, and Minkowski-p distances.
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
            ids.into_pyarray(py).to_object(py),
            dists.into_pyarray(py).to_object(py),
        ))
    }

    /// Batch-search k nearest neighbors for each row in an (N×dim) array.
    pub fn search_batch(
        &self,
        py: Python,
        data: PyReadonlyArray2<f32>,
        k: usize,
    ) -> PyResult<(PyObject, PyObject)> {
        let arr = data.as_array();
        let n = arr.nrows();

        // Release the GIL around the parallel batch:
        let results: Vec<(Vec<i64>, Vec<f32>)> = py.allow_threads(|| {
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let row = arr.row(i);
                    let q: Vec<f32> = row.to_vec();
                    let q_sq = q.iter().map(|x| x * x).sum::<f32>();
                    // safe unwrap: dims validated
                    self.inner_search(&q, q_sq, k).unwrap()
                })
                .collect::<Vec<_>>()
        });

        // Flatten the results
        let mut all_ids = Vec::with_capacity(n * k);
        let mut all_dists = Vec::with_capacity(n * k);
        for (ids, dists) in results {
            all_ids.extend(ids);
            all_dists.extend(dists);
        }

        // Build (n × k) ndarrays
        let ids_arr: Array2<i64> = Array2::from_shape_vec((n, k), all_ids)
            .map_err(|e| RustAnnError::py_err("Reshape Error",format!("Reshape ids failed: {}", e)))?;
        let dists_arr: Array2<f32> = Array2::from_shape_vec((n, k), all_dists)
            .map_err(|e| RustAnnError::py_err("Reshape Error",format!("Reshape dists failed: {}", e)))?;

        Ok((
            ids_arr.into_pyarray(py).to_object(py),
            dists_arr.into_pyarray(py).to_object(py),
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
}

impl AnnIndex {
    /// Core search logic covering L2, Cosine, L1 (Manhattan), L∞ (Chebyshev), and Lₚ.
    fn inner_search(&self, q: &[f32], q_sq: f32, k: usize) -> PyResult<(Vec<i64>, Vec<f32>)> {
        if q.len() != self.dim {
            return Err(RustAnnError::py_err("Dimension Error",format!(
                "Expected dimension {}, got {}", self.dim, q.len()
            )));
        }

        let p_opt = self.minkowski_p;
        let mut results: Vec<(i64, f32)> = self.entries
            .par_iter()
            .map(|(id, vec, vec_sq)| {
                // dot only used by L2/Cosine
                let dot = vec.iter().zip(q.iter()).map(|(x, y)| x * y).sum::<f32>();

                let dist = if let Some(p) = p_opt {
                    // Minkowski-p: (∑ |x-y|^p)^(1/p)
                    let sum_p = vec.iter().zip(q.iter())
                        .map(|(x, y)| (x - y).abs().powf(p))
                        .sum::<f32>();
                    sum_p.powf(1.0 / p)
                } else {
                    match self.metric {
                        Distance::Euclidean => ((vec_sq + q_sq - 2.0 * dot).max(0.0)).sqrt(),
                        Distance::Cosine    => {
                            let denom = vec_sq.sqrt().max(1e-12) * q_sq.sqrt().max(1e-12);
                            (1.0 - (dot / denom)).max(0.0)
                        }
                        Distance::Manhattan => vec.iter().zip(q.iter())
                            .map(|(x, y)| (x - y).abs())
                            .sum::<f32>(),
                        Distance::Chebyshev => vec.iter().zip(q.iter())
                            .map(|(x, y)| (x - y).abs())
                            .fold(0.0, f32::max),
                    }
                };

                (*id, dist)
            })
            .collect();

        // Sort ascending by distance and keep top-k
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);

        // Split into IDs and distances
        let ids   = results.iter().map(|(i, _)| *i).collect();
        let dists = results.iter().map(|(_, d)| *d).collect();
        Ok((ids, dists))
    }
}
