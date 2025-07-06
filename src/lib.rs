//! 🚀 rust_annie - Blazingly fast Approximate Nearest Neighbors in Rust
//!
//! This library provides efficient implementations of nearest neighbor search algorithms.
//! 
//! ## Features
//! - Multiple distance metrics (Euclidean, Cosine, Manhattan, Chebyshev)
//! - Brute-force and HNSW backends
//! - Thread-safe indexes
//! - GPU acceleration support
//! - Filtered search capabilities
//!
//! ## Quick Start
//! ```python
//! import numpy as np
//! from rust_annie import AnnIndex, Distance
//!
//! # Create index
//! dim = 128
//! index = AnnIndex(dim, Distance.EUCLIDEAN)
//!
//! # Add data
//! data = np.random.rand(1000, dim).astype(np.float32)
//! ids = np.arange(1000, dtype=np.int64)
//! index.add(data, ids)
//!
//! # Search
//! query = np.random.rand(dim).astype(np.float32)
//! neighbor_ids, distances = index.search(query, k=5)
//! ```
//!
//! ## Advanced Usage
//! ```python
//! # Thread-safe index
//! from rust_annie import ThreadSafeAnnIndex
//! ts_index = ThreadSafeAnnIndex(dim, Distance.COSINE)
//!
//! # HNSW index
//! from rust_annie import PyHnswIndex
//! hnsw_index = PyHnswIndex(dims=128)
//!
//! # Filtered search
//! def even_ids(id: int) -> bool:
//!     return id % 2 == 0
//!     
//! ids, dists = index.search_filter_py(query, k=5, filter_fn=even_ids)
//! ```
//!
//! ## Saving/Loading
//! ```python
//! index.save("my_index.bin")
//! loaded = AnnIndex.load("my_index.bin")
//! ```

mod utils;
pub mod index;
mod storage;
pub mod metrics;
mod errors;
mod concurrency;

mod backend;
pub mod hnsw_index;
mod index_enum;
mod filters;

// Add GPU module
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub mod gpu;

use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use crate::backend::AnnBackend;
use crate::index::AnnIndex;
use crate::metrics::Distance;
use crate::concurrency::ThreadSafeAnnIndex;
use crate::hnsw_index::HnswIndex;

#[pyclass]
pub struct PyHnswIndex {
    inner: HnswIndex,
}

#[pymethods]
impl PyHnswIndex {
    #[new]
    fn new(dims: usize) -> Self {
        PyHnswIndex {
            inner: HnswIndex::new(dims, Distance::Euclidean),
        }
    }

    fn add_item(&mut self, item: Vec<f32>) {
        self.inner.add_item(item);
    }

    fn add(&mut self, py: Python, data: PyReadonlyArray2<f32>, ids: PyReadonlyArray1<i64>) -> PyResult<()> {
        if !data.dtype().is_equiv_to(numpy::dtype::<f32>(py)) {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Input data must be of type f32"));
        }

        if !ids.dtype().is_equiv_to(numpy::dtype::<i64>(py)) {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "ids array must be of type i64",
            ));
        }
        
        let dims = self.inner.dims();
        let shape = data.shape();
        if shape.len() != 2 || shape[1] != dims {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Input data must be of shape (n, {})", dims),
            ));
        }

        let data_slice = data.as_slice()?;
        let ids_slice = ids.as_slice()?;
        let n_vectors = shape[0];

        if ids_slice.len() != n_vectors {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "ids length must match number of vectors",
            ));
        }

        for (i, vector) in data_slice.chunks_exact(dims).enumerate() {
            self.inner.insert(vector, ids_slice[i]);
        }
        Ok(())
    }

    fn build(&mut self) {
        self.inner.build();
    }

    fn search(&self, vector: Vec<f32>, k: usize) -> Vec<usize> {
        self.inner.search(&vector, k)
    }

    fn save(&self, path: String) {
        self.inner.save(&path);
    }

    #[staticmethod]
    fn load(_path: String) -> PyResult<Self> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "load() is not supported in hnsw-rs v0.3.2",
        ))
    }
}

#[pymodule]
fn rust_annie(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<AnnIndex>()?;
    m.add_class::<Distance>()?;
    m.add_class::<ThreadSafeAnnIndex>()?;
    m.add_class::<PyHnswIndex>()?;
    Ok(())
}
