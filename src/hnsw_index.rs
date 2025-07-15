use std::fs::File;
use std::io::{BufReader, BufWriter};
use bincode;
use serde::{Serialize, Deserialize};
use hnsw_rs::prelude::*;
use crate::backend::AnnBackend;
use crate::metrics::Distance;
use crate::utils::validate_path; // Import path validation utility

#[derive(Serialize, Deserialize)]
struct HnswIndexData {
    dims: usize,
    user_ids: Vec<i64>,
}

pub struct HnswIndex {
    index: Hnsw<'static, f32, DistL2>,
    dims: usize,
    user_ids: Vec<i64>, // Maps internal ID → user ID
}

impl AnnBackend for HnswIndex {
    fn new(dims: usize, _distance: Distance) -> Self {
        let index = Hnsw::new(
            16,     // M
            10_000, // max elements
            16,     // ef_construction
            200,    // ef_search
            DistL2 {},
        );
        HnswIndex {
            index,
            dims,
            user_ids: Vec::new(),
        }
    }

    fn add(&mut self, item: Vec<f32>, user_id: i64) {
        let internal_id = self.user_ids.len();
        self.index.insert((&item, internal_id));
        self.user_ids.push(user_id);
        self.vectors.push(item);
    }

    fn build(&mut self) {
        // No-op: HNSW builds during insertion
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

        HnswIndex {
            index,
            dims: data.dims,
            user_ids: data.user_ids,
        }
    }
}

impl HnswIndex {
    pub fn insert(&mut self, item: &[f32], user_id: i64) {
        let internal_id = self.user_ids.len();
        self.index.insert((item, internal_id));
        self.user_ids.push(user_id);
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