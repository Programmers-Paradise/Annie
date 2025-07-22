use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};

use bincode;
use serde::{Serialize, Deserialize};
use crate::metrics::Distance;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use crate::errors::RustAnnError;
use crate::index::AnnIndex;
use std::sync::atomic::AtomicU64;

#[derive(Serialize, Deserialize)]
struct SerializedAnnIndex {
    dim: usize,
    metric: Distance,
    minkowski_p: Option<f32>,
    entries: Vec<Option<(i64, Vec<f32>, f32)>>,
    deleted_count: usize,
    max_deleted_ratio: f32,
    version: u64,
}

/// Serialize and write the given index to `path` using bincode.
///
/// Returns a Python IOError on failure.
pub fn save_index(idx: &AnnIndex, path: &str) -> Result<(), RustAnnError> {
    let _serialized = SerializedAnnIndex {
        dim: idx.dim,
        metric: idx.metric.clone(),
        minkowski_p: idx.minkowski_p,
        entries: idx.entries.iter().map(|e| e.as_ref().map(|(id, v, s)| (*id, v.clone(), *s))).collect(),
        deleted_count: idx.deleted_count,
        max_deleted_ratio: idx.max_deleted_ratio,
        version: idx.version(),
    };
    let path = Path::new(path);
    let file = File::create(path)
        .map_err(|e| RustAnnError::io_err(format!("Failed to create file {}: {}", path.display(), e)))?;
    let writer = BufWriter::new(file);
    bincode::serialize_into(writer, idx)
        .map_err(|e| RustAnnError::io_err(format!("Serialization error: {}", e)))?;
    Ok(())
}

/// Read and deserialize an `AnnIndex` from `path` using bincode.
///
/// Returns a Python IOError on failure.
pub fn load_index(path: &str) -> Result<AnnIndex, RustAnnError> {
    let path = Path::new(path);
    let file = File::open(path)
        .map_err(|e| RustAnnError::io_err(format!("Failed to open file {}: {}", path.display(), e)))?;
    let reader = BufReader::new(file);
    // First read into the serialized representation
    let serialized: SerializedAnnIndex = bincode::deserialize_from(reader)
        .map_err(|e| RustAnnError::io_err(format!("Deserialization error: {}", e)))?;
    Ok(AnnIndex {
        dim: serialized.dim,
        metric: serialized.metric,
        minkowski_p: serialized.minkowski_p,
        entries: serialized.entries,
        deleted_count: serialized.deleted_count,
        max_deleted_ratio: serialized.max_deleted_ratio,
        metrics: None,
        boolean_filters: Mutex::new(HashMap::new()),
        version: Arc::new(AtomicU64::new(serialized.version)),
    })
}
