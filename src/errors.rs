// src/errors.rs
use std::fmt;
use std::sync::PoisonError;
use pyo3::exceptions::{PyException, PyIOError,PyRuntimeError, PyValueError};
use pyo3::PyErr;

/// A simple error type for the ANN library, used to convert Rust errors into Python exceptions.
#[derive(Debug)]
pub struct RustAnnError(pub String);

#[derive(Debug)]
pub enum BackendCreationError {
    UnsupportedBackend(String),
}

impl RustAnnError {
    /// Create a generic Python exception (`Exception`) with the error type and error message.
    pub fn py_err(type_name: impl Into<String>, detail: impl Into<String>) -> PyErr {
        let safe_type = type_name.into().replace(['\n', '\r', '[', ']'], " ");
        let safe_detail = detail.into().replace(['\n', '\r'], " ");
        let msg = format!("RustAnnError [{}]: {}", safe_type, safe_detail);
        PyException::new_err(msg)
    }

    /// Create a RustAnnError wrapping an I/O error message.
    /// This is used internally in save/load to signal I/O or serialization failures.
    pub fn io_err(msg: impl Into<String>) -> RustAnnError {
        RustAnnError(msg.into())
    }

    /// Convert this RustAnnError into a Python `IOError` (`OSError`) exception.
    pub fn into_pyerr(self) -> PyErr {
        PyIOError::new_err(self.0)
    }
}

impl fmt::Display for RustAnnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

pub enum DistanceRegistryError {
    LockPoisoned,
    RegistryNotInitialized,
    PythonCallFailed(String),
    PythonConversionFailed(String),
    MetricNotFound(String),
}

impl fmt::Display for DistanceRegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LockPoisoned => write!(f, "Global lock poisoned"),
            Self::RegistryNotInitialized => write!(f, "Distance registry not initialized"),
            Self::PythonCallFailed(e) => write!(f, "Python call failed: {}", e),
            Self::PythonConversionFailed(e) => write!(f, "Python value conversion failed: {}", e),
            Self::MetricNotFound(name) => write!(f, "Metric '{}' not found", name),
        }
    }
}

impl fmt::Display for BackendCreationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendCreationError::UnsupportedBackend(name) => {
                write!(f, "Unsupported backend: '{}'.", name)
            }
        }
    }
}

impl std::error::Error for BackendCreationError {}

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
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err.0)
    }
}

impl<T> From<PoisonError<T>> for DistanceRegistryError {
    fn from(_: PoisonError<T>) -> Self {
        Self::LockPoisoned
    }
}
