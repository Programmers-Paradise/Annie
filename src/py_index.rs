use pyo3::prelude::*;
use numpy::PyReadonlyArray1;
use crate::index_enum::Index;
use crate::metrics::Distance;

/// A unified Python-facing ANN Index supporting "brute" and "hnsw" backends.
#[pyclass(name = "Index")]
pub struct PyIndex {
    index: Index,
}

#[pymethods]
impl PyIndex {
    /// Constructor: Index(kind="brute" | "hnsw", dim=128, metric=Distance.EUCLIDEAN)
    #[new]
    #[pyo3(signature = (kind, dim, metric))]
    fn new(kind: &str, dim: usize, metric: Distance) -> PyResult<Self> {
        let index = match kind.to_lowercase().as_str() {
            "brute" => Index::BruteForce(crate::AnnIndex::new(dim, metric)?),
            "hnsw" => {
                // Default config: same values you hardcoded earlier
                let config = crate::hnsw_index::HnswConfig::default();
                let hnsw = crate::HnswIndex::new_with_config(dim, config)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Index::Hnsw(hnsw)
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown backend '{}'. Use 'brute' or 'hnsw'.", kind
                )));
            }
        };
        Ok(Self { index })
    }

    fn add_item(&mut self, item: Vec<f32>) {
        self.index.add_item(item);
    }

    fn build(&mut self) {
        self.index.build();
    }

    fn search(
        &self,
        py: Python<'_>,
        vector: PyReadonlyArray1<f32>,
        k: usize,
    ) -> PyResult<(PyObject, PyObject)> {
        self.index.search(py, vector, k)
    }
}
