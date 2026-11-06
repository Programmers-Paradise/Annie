// src/errors.rs
use pyo3::exceptions::{
    PyIOError, PyMemoryError, PyNotImplementedError, PyRuntimeError, PyTypeError, PyValueError,
};
use pyo3::PyErr;
use std::sync::PoisonError;
use thiserror::Error;

/// A standardized error type for the ANN library, with context and chaining.
#[derive(Debug, Error)]
pub enum RustAnnError {
    #[error("Lock poisoned")]
    LockPoisoned,
    #[error("{0}")]
    Message(String),
    #[error("I/O error: {0}")]
    Io(String),
    #[error("Callback error: {0}")]
    Callback(String),
    #[error("Duplicate IDs: {0}")]
    DuplicateIds(String),
    #[error("Dimension error: {0}")]
    Dimension(String),
    #[error("Allocation error: {0}")]
    Allocation(String),
    #[error("Empty index")]
    EmptyIndex,
    #[error("Reshape error: {0}")]
    Reshape(String),
    #[error("Minkowski error: {0}")]
    Minkowski(String),
    #[error("Other error: {0}")]
    Other(String),
}

#[derive(Debug, Error)]
pub enum BackendCreationError {
    #[error("Unsupported backend: '{0}'.")]
    UnsupportedBackend(String),
}

impl RustAnnError {
    /// Create a Python exception for the given error category.
    ///
    /// The category is mapped onto the most specific built-in Python exception
    /// (`ValueError` by default; `OSError` for I/O and path problems,
    /// `NotImplementedError`, `RuntimeError`, `TypeError`, `MemoryError`), so
    /// callers can catch errors idiomatically. All of these subclass
    /// `Exception`, so `except Exception` keeps working.
    pub fn py_err(type_name: impl Into<String>, detail: impl Into<String>) -> PyErr {
        let type_name = type_name.into();
        let safe_type = type_name.replace(['\n', '\r', '[', ']'], " ");
        let safe_detail = detail.into().replace(['\n', '\r'], " ");
        let msg = format!("RustAnnError [{}]: {}", safe_type, safe_detail);
        match type_name.as_str() {
            "NotImplemented" => PyNotImplementedError::new_err(msg),
            "IOError" | "InvalidPath" => PyIOError::new_err(msg),
            "LockError" | "ConcurrentModification" | "Reshape Error" => {
                PyRuntimeError::new_err(msg)
            }
            "TypeError" => PyTypeError::new_err(msg),
            "Excessive Allocation" => PyMemoryError::new_err(msg),
            _ => PyValueError::new_err(msg),
        }
    }
    pub fn io_err(msg: impl Into<String>) -> RustAnnError {
        RustAnnError::Io(msg.into())
    }
    pub fn into_pyerr(self) -> PyErr {
        match self {
            RustAnnError::Io(msg) => PyIOError::new_err(msg),
            RustAnnError::Callback(msg) => PyRuntimeError::new_err(msg),
            RustAnnError::DuplicateIds(msg) => PyValueError::new_err(msg),
            RustAnnError::Dimension(msg) => PyValueError::new_err(msg),
            RustAnnError::Allocation(msg) => PyRuntimeError::new_err(msg),
            RustAnnError::EmptyIndex => PyValueError::new_err("Index is empty"),
            RustAnnError::Reshape(msg) => PyValueError::new_err(msg),
            RustAnnError::Minkowski(msg) => PyValueError::new_err(msg),
            RustAnnError::Message(msg) => PyRuntimeError::new_err(msg),
            RustAnnError::Other(msg) => PyRuntimeError::new_err(msg),
            RustAnnError::LockPoisoned => PyRuntimeError::new_err("Lock poisoned"),
        }
    }
}

// Display is derived by thiserror

#[derive(Debug, Error)]
pub enum DistanceRegistryError {
    #[error("Global lock poisoned")]
    LockPoisoned,
    #[error("Distance registry not initialized")]
    RegistryNotInitialized,
    #[error("Python call failed: {0}")]
    PythonCallFailed(String),
    #[error("Python value conversion failed: {0}")]
    PythonConversionFailed(String),
    #[error("Metric '{0}' not found")]
    MetricNotFound(String),
}

impl From<BackendCreationError> for PyErr {
    fn from(err: BackendCreationError) -> Self {
        PyValueError::new_err(err.to_string())
    }
}

impl From<DistanceRegistryError> for PyErr {
    fn from(e: DistanceRegistryError) -> PyErr {
        PyRuntimeError::new_err(e.to_string())
    }
}

impl From<RustAnnError> for PyErr {
    fn from(err: RustAnnError) -> Self {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err.to_string())
    }
}

impl<T> From<PoisonError<T>> for DistanceRegistryError {
    fn from(_: PoisonError<T>) -> Self {
        Self::LockPoisoned
    }
}
