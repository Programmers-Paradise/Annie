// src/path_validation.rs
//! Validation of user-supplied file paths for `save` / `load`.
//!
//! `rust_annie` is a library: the caller decides where indexes live, exactly
//! as with `numpy.save`, `pickle.dump` or `faiss.write_index`. We therefore do
//! **not** restrict paths to the current working directory. Applications that
//! accept paths from untrusted users must sandbox them themselves.
//!
//! What we do reject are paths that are malformed and would otherwise produce
//! confusing OS-level errors: empty paths, paths containing NUL bytes or other
//! control characters, and paths that name an existing directory.

use crate::errors::RustAnnError;
use pyo3::PyResult;
use std::path::PathBuf;

/// Validate a path supplied for saving or loading an index.
///
/// Returns the path as a [`PathBuf`] on success.
///
/// # Examples
/// ```
/// use rust_annie::path_validation::validate_path_secure;
///
/// assert!(validate_path_secure("model.bin").is_ok());
/// assert!(validate_path_secure("./data/index.bin").is_ok());
/// assert!(validate_path_secure("").is_err());
/// assert!(validate_path_secure("bad\0name.bin").is_err());
/// ```
pub fn validate_path_secure(path: &str) -> PyResult<PathBuf> {
    if path.is_empty() {
        return Err(RustAnnError::py_err(
            "InvalidPath",
            "Path must not be empty",
        ));
    }
    if path.chars().any(|c| c.is_control()) {
        return Err(RustAnnError::py_err(
            "InvalidPath",
            "Path contains NUL bytes or control characters",
        ));
    }
    let path_buf = PathBuf::from(path);
    if path_buf.is_dir() {
        return Err(RustAnnError::py_err(
            "InvalidPath",
            format!("Path '{}' is a directory, expected a file path", path),
        ));
    }
    Ok(path_buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_relative_and_absolute_paths() {
        assert!(validate_path_secure("test.bin").is_ok());
        assert!(validate_path_secure("./data/model.bin").is_ok());
        let abs = std::env::temp_dir().join("rust_annie_test.bin");
        assert!(validate_path_secure(abs.to_str().unwrap()).is_ok());
    }

    #[test]
    fn rejects_malformed_paths() {
        assert!(validate_path_secure("").is_err());
        assert!(validate_path_secure("test.bin\0").is_err());
        assert!(validate_path_secure("test\x01.bin").is_err());
        assert!(validate_path_secure("line\nbreak.bin").is_err());
    }

    #[test]
    fn rejects_directories() {
        let dir = std::env::temp_dir();
        assert!(validate_path_secure(dir.to_str().unwrap()).is_err());
    }
}
