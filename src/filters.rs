use bit_vec::BitVec;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Clone, Serialize, Deserialize)]
pub enum FilterType {
    IdRange(i64, i64),
    IdSet(HashSet<i64>),
    Boolean(BitVec),
    And(Vec<Filter>),
    Or(Vec<Filter>),
    Not(Box<Filter>),
}

/// Predicate restricting which IDs a search may return.
///
/// Build filters with the static constructors (`id_range`, `id_set`,
/// `boolean`) and combine them with `and_`, `or_` and `not_`. A plain Python
/// callable `f(id) -> bool` can also be passed wherever a filter is accepted.
#[pyclass(module = "rust_annie")]
#[derive(Clone, Serialize, Deserialize)]
pub struct Filter {
    inner: FilterType,
}

#[pymethods]
impl Filter {
    /// Accept IDs in the inclusive range `[min, max]`.
    #[staticmethod]
    pub fn id_range(min: i64, max: i64) -> Self {
        Filter {
            inner: FilterType::IdRange(min, max),
        }
    }

    /// Accept only the given IDs.
    #[staticmethod]
    pub fn id_set(ids: Vec<i64>) -> Self {
        Filter {
            inner: FilterType::IdSet(ids.into_iter().collect()),
        }
    }

    /// Accept entries whose storage position `i` has `bits[i] == True`.
    /// Positions follow insertion order and shift after `compact()`.
    #[staticmethod]
    pub fn boolean(bits: Vec<bool>) -> Self {
        let mut bv = BitVec::from_elem(bits.len(), false);
        for (i, &bit) in bits.iter().enumerate() {
            bv.set(i, bit);
        }
        Filter {
            inner: FilterType::Boolean(bv),
        }
    }

    /// Accept IDs accepted by every filter.
    #[staticmethod]
    #[pyo3(name = "and_")]
    pub fn and(filters: Vec<Filter>) -> Self {
        Filter {
            inner: FilterType::And(filters),
        }
    }

    /// Accept IDs accepted by any filter.
    #[staticmethod]
    #[pyo3(name = "or_")]
    pub fn or(filters: Vec<Filter>) -> Self {
        Filter {
            inner: FilterType::Or(filters),
        }
    }

    /// Accept IDs rejected by `filter`.
    #[staticmethod]
    #[pyo3(name = "not_")]
    #[allow(clippy::should_implement_trait)]
    pub fn not(filter: Filter) -> Self {
        Filter {
            inner: FilterType::Not(Box::new(filter)),
        }
    }

    /// Whether this filter accepts the entry with the given ID and storage
    /// position. `index` is only consulted by `Filter.boolean`.
    pub fn accepts(&self, id: i64, index: usize) -> bool {
        match &self.inner {
            FilterType::IdRange(min, max) => id >= *min && id <= *max,
            FilterType::IdSet(ids) => ids.contains(&id),
            FilterType::Boolean(bits) => bits.get(index).unwrap_or(false),
            FilterType::And(filters) => filters.iter().all(|f| f.accepts(id, index)),
            FilterType::Or(filters) => filters.iter().any(|f| f.accepts(id, index)),
            FilterType::Not(filter) => !filter.accepts(id, index),
        }
    }
}
