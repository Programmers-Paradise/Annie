use hnsw_rs::prelude::*;
use crate::backends::ann_backend::AnnBackend;
use crate::metrics::Distance;
use rust_annie_macros::py_annindex;

/// HNSW backend implementation.
/// For now, only supports Euclidean (L2) distance.
#[py_annindex(backend = "HNSW", distance = "Euclidean")]
pub struct HnswIndex {
    index: Hnsw<'static, f32, DistL2>,
    dims: usize,
}

impl HnswIndex {
    pub fn new(dims: usize, _distance: Distance) -> Self {
        let index = Hnsw::new(
            16,     // M: number of bi-directional links
            10_000, // max elements
            16,     // ef_construction
            200,    // ef_search
            DistL2 {},
        );
        Self { index, dims }
    }
}

impl AnnBackend for HnswIndex {
    fn add(&mut self, vector: Vec<f32>) {
        let id = self.index.get_nb_point();
        self.index.insert((&vector, id));
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        self.index
            .search(query, k, 50)
            .into_iter()
            .map(|n| (n.d_id as usize, n.distance))
            .collect()
    }

    fn len(&self) -> usize {
        self.index.get_nb_point() as usize
    }
}
