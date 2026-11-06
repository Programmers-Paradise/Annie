//! `rust_annie`: exact and approximate nearest-neighbour search for Python,
//! implemented in Rust.
//!
//! The Python package re-exports everything registered in [`_rust_annie`].

pub mod concurrency;
pub mod distance_registry;
mod errors;
pub mod filters;
pub mod hnsw_index;
pub mod index;
pub mod metrics;
pub mod monitoring;
pub mod path_validation;
pub mod py_index;
pub mod search;
pub mod storage;

/// Experimental GPU kernels (Rust API only; not used by the Python bindings).
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub mod gpu;

pub use crate::concurrency::ThreadSafeAnnIndex;
pub use crate::errors::RustAnnError;
pub use crate::filters::Filter;
pub use crate::hnsw_index::{HnswConfig, HnswIndex};
pub use crate::index::AnnIndex;
pub use crate::metrics::Distance;

use pyo3::prelude::*;

#[pymodule]
fn _rust_annie(m: &Bound<'_, PyModule>) -> PyResult<()> {
    distance_registry::init_distance_registry();

    m.add_class::<AnnIndex>()?;
    m.add_class::<ThreadSafeAnnIndex>()?;
    m.add_class::<hnsw_index::PyHnswIndex>()?;
    m.add_class::<hnsw_index::PyHnswConfig>()?;
    m.add_class::<py_index::PyIndex>()?;
    m.add_class::<Distance>()?;
    m.add_class::<Filter>()?;
    m.add_class::<monitoring::PyMetricsCollector>()?;
    m.add_function(wrap_pyfunction!(distance_registry::register_metric, m)?)?;
    m.add_function(wrap_pyfunction!(distance_registry::list_metrics, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
