// src/storage.rs

use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};

use crate::errors::RustAnnError;
use crate::index::AnnIndex;

/// Serialize and write the given index to `path` using bincode.
///
/// Returns a Python IOError on failure.
pub fn save_index(idx: &AnnIndex, path: &str) -> Result<(), RustAnnError> {
    let path = Path::new(path);
    let file = File::create(path)
        .map_err(|e| RustAnnError::io_err(format!("Failed to create file {}: {e}", path.display())))?;
    let writer = BufWriter::new(file);
    bincode::serialize_into(writer, idx)
        .map_err(|e| RustAnnError::io_err(format!("Serialization error: {e}")))?;
    Ok(())
}

/// Read and deserialize an `AnnIndex` from `path` using bincode.
///
/// Returns a Python IOError on failure.
pub fn load_index(path: &str) -> Result<AnnIndex, RustAnnError> {
    let path = Path::new(path);
    let file = File::open(path)
        .map_err(|e| RustAnnError::io_err(format!("Failed to open file {}: {e}", path.display())))?;
    let reader = BufReader::new(file);
    let idx: AnnIndex = bincode::deserialize_from(reader)
        .map_err(|e| RustAnnError::io_err(format!("Deserialization error: {e}")))?;
    Ok(idx)
}
