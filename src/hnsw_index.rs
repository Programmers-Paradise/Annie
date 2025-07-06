use hnsw_rs::prelude::*;
use crate::backend::AnnBackend;
use crate::metrics::Distance;

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

    fn add_item(&mut self, item: Vec<f32>) {
        let internal_id = self.user_ids.len();
        self.index.insert((&item, internal_id));
        self.user_ids.push(internal_id as i64); // default internal ID as user ID
    }

    fn build(&mut self) {
        // No-op: HNSW builds during insertion
    }

    fn search(&self, vector: &[f32], k: usize) -> Vec<usize> {
        let results = self.index.search(vector, k, 50);
        let mut ids = Vec::with_capacity(results.len());
        for n in results {
            ids.push(n.d_id);
        }
        ids
    }

    fn save(&self, _path: &str) {
        unimplemented!("HNSW save not implemented yet");
    }

    fn load(_path: &str) -> Self {
        unimplemented!("HNSW load not implemented yet");
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
