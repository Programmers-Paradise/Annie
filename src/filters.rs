use pyo3::prelude::*;
use pyo3::PyObject;
use std::collections::HashSet;
use std::sync::Arc;
use serde::{Serialize, Deserialize};
use bit_vec::BitVec;

#[derive(Serialize, Deserialize)]
pub enum FilterType {
    IdRange(i64, i64),
    IdSet(HashSet<i64>),
    Boolean(BitVec),
    And(Vec<Filter>),
    Or(Vec<Filter>),
    Not(Box<Filter>),
    // PythonCallback variant must never be invoked directly; if implemented, ensure strict sandboxing and input validation to prevent arbitrary code execution.
}

#[pyclass]
#[derive(Clone, Serialize, Deserialize)]
pub struct Filter {
    inner: FilterType,
}

#[pymethods]
impl Filter {
    #[staticmethod]
    pub fn id_range(min: i64, max: i64) -> Self {
        Filter { inner: FilterType::IdRange(min, max) }
    }

    #[staticmethod]
    pub fn id_set(ids: Vec<i64>) -> Self {
        Filter { inner: FilterType::IdSet(ids.into_iter().collect()) }
    }

    #[staticmethod]
    pub fn boolean(bits: Vec<bool>) -> Self {
        let mut bv = BitVec::from_elem(bits.len(), false);
        for (i, &bit) in bits.iter().enumerate() {
            bv.set(i, bit);
        }
        Filter { FilterType::Boolean(bv) }
    }

    #[staticmethod]
    pub fn and(filters: Vec<Filter>) -> Self {
        Filter { FilterType::And(filters) }
    }

    #[staticmethod]
    pub fn or(filters: Vec<Filter>) -> Self {
        Filter { FilterType::Or(filters) }
    }

    #[staticmethod]
    pub fn not(filter: Filter) -> Self {
        Filter { FilterType::Not(Box::new(filter)) }
    }

    #[staticmethod]
    pub fn from_py_callable(_callback: PyObject) -> Self {
        Filter { FilterType::PythonCallback }
    }

    pub fn accepts(&self, id: i64, index: usize) -> bool {
        match &*self.inner {
            FilterType::IdRange(min, max) => id >= *min && id <= *max,
            FilterType::IdSet(ids) => ids.contains(&id),
            FilterType::Boolean(bits) => unsafe {
                // SAFETY: Caller must ensure index < bits.len()
                bits.get_unchecked(index)
            },
            FilterType::And(filters) => filters.iter().all(|f| f.accepts(id, index)),
            FilterType::Or(filters) => filters.iter().any(|f| f.accepts(id, index)),
            FilterType::Not(filter) => !filter.accepts(id, index),
            _ => false, // Callbacks handled separately
        }
    }
}

#[allow(dead_code)]
pub struct PythonFilter {
    callback: PyObject,
}

#[allow(dead_code)]
impl PythonFilter {
    pub fn new(callback: PyObject) -> Self {
        PythonFilter { callback }
    }

    /// Check if the given ID passes the Python filter
    pub fn accepts(&self, py: Python<'_>, id: i64) -> PyResult<bool> {
        let result = self.callback.call1(py, (id,))?;
        result.extract::<bool>(py)
    }
}
