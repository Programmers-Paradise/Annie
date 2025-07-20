use pyo3::prelude::*;
use pyo3::PyObject;
use std::collections::HashSet;
use serde::{Serialize, Deserialize};
use bit_vec::BitVec;

#[derive(Clone, Serialize, Deserialize)]
pub enum FilterType {
    IdRange(i64, i64),
    IdSet(HashSet<i64>),
    Boolean(BitVec),
    And(Vec<Filter>),
    Or(Vec<Filter>),
    Not(Box<Filter>),
    PythonCallback,
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
        Filter { inner: FilterType::Boolean(bv) }
    }

    #[staticmethod]
    pub fn and(filters: Vec<Filter>) -> Self {
        Filter { inner: FilterType::And(filters) }
    }

    #[staticmethod]
    pub fn or(filters: Vec<Filter>) -> Self {
        Filter { inner: FilterType::Or(filters) }
    }

    #[staticmethod]
    pub fn not(filter: Filter) -> Self {
        Filter { inner: FilterType::Not(Box::new(filter)) }
    }

    #[staticmethod]
    pub fn from_py_callable(_callback: PyObject) -> Self {
        Filter { inner: FilterType::PythonCallback }
    }

    pub fn accepts(&self, id: i64, index: usize) -> bool {
        match &self.inner {
            FilterType::IdRange(min, max) => id >= *min && id <= *max,
            FilterType::IdSet(ids) => ids.contains(&id),
            FilterType::Boolean(bits) => {
                if index < bits.len() {
                    bits.get(index).unwrap_or(false)
                } else {
                    false
                }
            }
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
