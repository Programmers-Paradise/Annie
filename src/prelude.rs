// src/prelude.rs

//! A “prelude” to import the most common types and traits in one go.

pub use crate::index::AnnIndex;
pub use crate::metrics::Distance;
pub use crate::storage::{save_index, load_index};
pub use crate::concurrency::ThreadSafeIndex;
pub use crate::errors::RustAnnError;
