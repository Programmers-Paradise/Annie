// src/errors.rs

use pyo3::exceptions::{PyException, PyIOError};
use pyo3::PyErr;

/// A simple error type for the ANN library, used to convert Rust errors into Python exceptions.
#[derive(Debug)]
pub struct RustAnnError(pub String);

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
