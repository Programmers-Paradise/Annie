use std::fs::File;
use std::io::{BufReader, BufWriter};
use bincode;
use serde::{Serialize, Deserialize};
use hnsw_rs::prelude::*;
use crate::backend::AnnBackend;
use crate::metrics::Distance;

#[derive(Serialize, Deserialize)]
struct HnswIndexData {
    dims: usize,
    m: u16,
    max_elements: usize,
    ef_construction: usize,
    ef_search: usize,
    user_ids: Vec<i64>,
    vectors: Vec<Vec<f32>>,
}

pub struct HnswIndex {
    index: Hnsw<'static, f32, DistL2>,
    dims: usize,
    user_ids: Vec<i64>, // Maps internal ID → user ID
    vectors: Vec<Vec<f32>>,
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
            vectors: Vec::new(),
        }
    }

    fn add_item(&mut self, item: Vec<f32>) {
        let internal_id = self.user_ids.len();
        self.index.insert((&item, internal_id));
        self.user_ids.push(internal_id as i64); // default internal ID as user ID
        self.vectors.push(item);
    }

    fn build(&mut self) {
        // No-op: HNSW builds during insertion
    }

    fn search(&self, vector: &[f32], k: usize) -> Vec<usize> {
        let results = self.index.search(vector, k, 50);
        results.into_iter().map(|n| n.d_id).collect()
    }

    fn save(&self, _path: &str) {
        let data = HnswIndexData {
            dims: self.dims,
            m: self.index.get_m(),
            max_elements: self.index.get_max_elements(),
            ef_construction: self.index.get_ef_construction(),
            ef_search: self.index.get_ef(),
            user_ids: self.user_ids.clone(),
            vectors: self.vectors.clone(),
        };

        let file = File::create(path).expect("Failed to create file");
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, &data).expect("Serialization failed");
    }

    fn load(_path: &str) -> Self {
        let file = File::open(path).expect("Failed to open file");
        let reader = BufReader::new(file);
        let data: HnswIndexData = bincode::deserialize_from(reader).expect("Deserialization failed");

        let mut index = Hnsw::new(
            data.m,
            data.max_elements,
            data.ef_construction,
            data.ef_search,
            DistL2 {},
        );

        for (internal_id, vec) in data.vectors.iter().enumerate() {
            index.insert((vec, internal_id));
        }

        HnswIndex {
            index,
            dims: data.dims,
            user_ids: data.user_ids,
            vectors: data.vectors,
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
