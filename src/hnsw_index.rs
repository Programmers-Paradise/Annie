//new added
use hnsw_rs::prelude::*;
use crate::backend::AnnBackend;
use crate::metrics::Distance;

pub struct HnswIndex {
    index: Hnsw<'static, f32, DistL2>,
    dims: usize,
}

impl AnnBackend for HnswIndex {
    fn new(dims: usize, _distance: Distance) -> Self {
        let index = Hnsw::new(
            16,
            10_000,
            16,
            200,
            DistL2 {},
        );
        HnswIndex { index, dims }
    }

    fn add_item(&mut self, item: Vec<f32>) {
        let id = self.index.get_nb_point();
        self.index.insert((&item, id));
    }

    fn build(&mut self) {}

    fn search(&self, vector: &[f32], k: usize) -> Vec<usize> {
        self.index
            .search(vector, k, 50)
            .iter()
            .map(|n| n.d_id)
            .collect()
    }

    fn save(&self, _path: &str) {
        panic!("save() not supported in hnsw-rs v0.3.2");
    }

    fn load(_path: &str) -> Self {
        panic!("load() not supported in hnsw-rs v0.3.2");
    }
}





