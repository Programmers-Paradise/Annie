use std::fs::File;
use std::io::{BufReader, BufWriter};
use bincode;
use serde::{Serialize, Deserialize};
use hnsw_rs::prelude::*;
use crate::backend::AnnBackend;
use crate::metrics::Distance;
use crate::utils::validate_path;
use crate::errors::RustAnnError;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HnswConfig {
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub max_elements: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 16,
            ef_search: 200,
            max_elements: 10_000,
        }
    }
}

impl HnswConfig {
    pub fn validate(&self) -> Result<(), RustAnnError> {
        if self.m == 0 {
            return Err(RustAnnError::io_err("`m` must be greater than 0"));
        }
        if self.ef_construction == 0 {
            return Err(RustAnnError::io_err("`ef_construction` must be greater than 0"));
        }
        if self.ef_search == 0 {
            return Err(RustAnnError::io_err("`ef_search` must be greater than 0"));
        }
        if self.max_elements == 0 {
            return Err(RustAnnError::io_err("`max_elements` must be greater than 0"));
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
struct HnswIndexData {
    dims: usize,
    user_ids: Vec<i64>,
    vectors: Vec<Vec<f32>>,
}

pub struct HnswIndex {
    index: Hnsw<'static, f32, DistL2>,
    dims: usize,
    user_ids: Vec<i64>,
    vectors: Vec<Vec<f32>>, // Added field
}

impl HnswIndex {
    pub fn new_with_config(dims: usize, config: HnswConfig) -> Result<Self, RustAnnError> {
        config.validate()?;

        let index = Hnsw::new(
            config.m,
            config.max_elements,
            config.ef_construction,
            config.ef_search,
            DistL2 {},
        );

        Ok(HnswIndex {
            index,
            dims,
            user_ids: Vec::new(),
            vectors: Vec::new(),
        })
    }

    pub fn insert(&mut self, item: &[f32], user_id: i64) {
        let internal_id = self.user_ids.len();
        self.index.insert((item, internal_id));
        self.user_ids.push(user_id);
        self.vectors.push(item.to_vec());
    }

    pub fn dims(&self) -> usize {
        self.dims
    }

    pub fn get_user_id(&self, internal_id: usize) -> i64 {
        if internal_id < self.user_ids.len() {
            self.user_ids[internal_id]
        } else {
            -1
        }
    }
}

impl AnnBackend for HnswIndex {
    fn new(dims: usize, _distance: Distance) -> Self {
        HnswIndex::new_with_config(dims, HnswConfig::default())
            .expect("Failed to create HNSW index with default config")
    }

    fn add_item(&mut self, item: Vec<f32>) {
        let internal_id = self.user_ids.len() as i64;
        self.index.insert((&item, internal_id as usize));
        self.user_ids.push(internal_id);
        self.vectors.push(item); // Valid now
    }

    fn build(&mut self) {
        // No-op for HNSW
    }

    fn search(&self, vector: &[f32], k: usize) -> Vec<usize> {
        let results = self.index.search(vector, k, 50);
        results.into_iter().map(|n| n.d_id).collect()
    }

    fn save(&self, path: &str) {
        let safe_path = validate_path(path).expect("Invalid or unsafe file path");

        let data = HnswIndexData {
            dims: self.dims,
            user_ids: self.user_ids.clone(),
            vectors: self.vectors.clone(),
        };

        let file = File::create(&safe_path).expect("Failed to create file");
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, &data).expect("Serialization failed");
    }

    fn load(path: &str) -> Self {
        let safe_path = validate_path(path).expect("Invalid or unsafe file path");

        let file = File::open(&safe_path).expect("Failed to open file");
        let reader = BufReader::new(file);
        let data: HnswIndexData = bincode::deserialize_from(reader).expect("Deserialization failed");

        let mut index = Hnsw::new(
            16,
            10_000,
            16,
            200,
            DistL2 {},
        );

        for (i, item) in data.vectors.iter().enumerate() {
            index.insert((item.as_slice(), i));
        }

        HnswIndex {
            index,
            dims: data.dims,
            user_ids: data.user_ids,
            vectors: data.vectors,
        }
    }
}

