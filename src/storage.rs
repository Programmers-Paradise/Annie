//! On-disk persistence for `rust_annie` indexes.
//!
//! Every file starts with an 8-byte magic header identifying the index kind,
//! followed by a little-endian `u32` format version and a bincode payload.
//! Deleted entries are never written, so a saved index is always compact.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::errors::RustAnnError;
use crate::hnsw_index::HnswConfig;
use crate::metrics::Distance;

/// Magic header for brute-force (`AnnIndex`) files.
pub const MAGIC_BRUTE: &[u8; 8] = b"ANNIE\0BF";
/// Magic header for HNSW (`PyHnswIndex`) files.
pub const MAGIC_HNSW: &[u8; 8] = b"ANNIE\0HN";
/// Current on-disk format version.
pub const FORMAT_VERSION: u32 = 1;

/// Which kind of index a file contains.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexKind {
    BruteForce,
    Hnsw,
}

/// Serialized payload of a brute-force index.
#[derive(Serialize, Deserialize)]
pub struct BruteForceData {
    pub dim: usize,
    pub metric: Distance,
    pub max_deleted_ratio: f32,
    /// Live entries as `(id, vector)` pairs, in storage order.
    pub entries: Vec<(i64, Vec<f32>)>,
}

/// Serialized payload of an HNSW index. The graph itself is rebuilt on load.
#[derive(Serialize, Deserialize)]
pub struct HnswData {
    pub dim: usize,
    pub metric: Distance,
    pub config: HnswConfig,
    /// Live entries as `(id, vector)` pairs, in insertion order.
    pub entries: Vec<(i64, Vec<f32>)>,
}

fn io_err(action: &str, path: &Path, e: impl std::fmt::Display) -> RustAnnError {
    RustAnnError::io_err(format!("Failed to {} {}: {}", action, path.display(), e))
}

fn write_file<T: Serialize>(path: &Path, magic: &[u8; 8], payload: &T) -> Result<(), RustAnnError> {
    let file = File::create(path).map_err(|e| io_err("create", path, e))?;
    let mut writer = BufWriter::new(file);
    writer
        .write_all(magic)
        .map_err(|e| io_err("write", path, e))?;
    writer
        .write_all(&FORMAT_VERSION.to_le_bytes())
        .map_err(|e| io_err("write", path, e))?;
    bincode::serialize_into(&mut writer, payload)
        .map_err(|e| io_err("serialize index to", path, e))?;
    writer.flush().map_err(|e| io_err("write", path, e))?;
    Ok(())
}

fn read_header(reader: &mut impl Read, path: &Path) -> Result<IndexKind, RustAnnError> {
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic).map_err(|_| {
        RustAnnError::io_err(format!(
            "{} is not a rust_annie index file (file too short)",
            path.display()
        ))
    })?;
    let kind = match &magic {
        m if m == MAGIC_BRUTE => IndexKind::BruteForce,
        m if m == MAGIC_HNSW => IndexKind::Hnsw,
        _ => {
            return Err(RustAnnError::io_err(format!(
                "{} is not a rust_annie index file (bad header). Files written by \
                 rust_annie < 0.3 are not supported; rebuild and re-save the index.",
                path.display()
            )))
        }
    };
    let mut version = [0u8; 4];
    reader
        .read_exact(&mut version)
        .map_err(|e| io_err("read", path, e))?;
    let version = u32::from_le_bytes(version);
    if version != FORMAT_VERSION {
        return Err(RustAnnError::io_err(format!(
            "{} uses index format version {}, but this build supports version {}",
            path.display(),
            version,
            FORMAT_VERSION
        )));
    }
    Ok(kind)
}

fn read_file<T: DeserializeOwned>(path: &Path, expected: IndexKind) -> Result<T, RustAnnError> {
    let file = File::open(path).map_err(|e| io_err("open", path, e))?;
    let mut reader = BufReader::new(file);
    let kind = read_header(&mut reader, path)?;
    if kind != expected {
        return Err(RustAnnError::io_err(format!(
            "{} contains a {:?} index, expected {:?}",
            path.display(),
            kind,
            expected
        )));
    }
    bincode::deserialize_from(reader).map_err(|e| io_err("deserialize index from", path, e))
}

/// Peek at a file's header to find out which kind of index it holds.
pub fn detect_kind(path: &Path) -> Result<IndexKind, RustAnnError> {
    let file = File::open(path).map_err(|e| io_err("open", path, e))?;
    let mut reader = BufReader::new(file);
    read_header(&mut reader, path)
}

pub fn save_brute(path: &Path, data: &BruteForceData) -> Result<(), RustAnnError> {
    write_file(path, MAGIC_BRUTE, data)
}

pub fn load_brute(path: &Path) -> Result<BruteForceData, RustAnnError> {
    read_file(path, IndexKind::BruteForce)
}

pub fn save_hnsw(path: &Path, data: &HnswData) -> Result<(), RustAnnError> {
    write_file(path, MAGIC_HNSW, data)
}

pub fn load_hnsw(path: &Path) -> Result<HnswData, RustAnnError> {
    read_file(path, IndexKind::Hnsw)
}
