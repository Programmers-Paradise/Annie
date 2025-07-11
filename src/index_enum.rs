use crate::{AnnIndex, HnswIndex, backend::AnnBackend};
use pyo3::{Python, PyResult, PyAny, Py};
use numpy::PyReadonlyArray1;

pub enum Index {
    BruteForce(AnnIndex),
    Hnsw(HnswIndex),
}

impl Index {
    pub fn add_item(&mut self, item: Vec<f32>) {
        match self {
            Index::BruteForce(bf) => bf.add_item(item),
            Index::Hnsw(hnsw) => hnsw.add_item(item),
        }
    }

    pub fn build(&mut self) {
        match self {
            Index::BruteForce(bf) => bf.build(),
            Index::Hnsw(hnsw) => hnsw.build(),
        }
    }

    pub fn search(
        &self,
        py: Python<'_>,
        vector: PyReadonlyArray1<f32>,
        k: usize,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        match self {
            Index::BruteForce(bf) => bf.search(py, vector, k),
            Index::Hnsw(hnsw) => {
                let vec_slice = vector.as_slice()?;
                let results = hnsw.search(vec_slice, k);
                let ids = results.iter().map(|&id| id as i64).collect::<Vec<_>>();
                let distances = vec![0.0; ids.len()];
                let ids_py = numpy::PyArray1::from_vec(py, ids).to_owned();
                let dist_py = numpy::PyArray1::from_vec(py, distances).to_owned();
                Ok((ids_py.into(), dist_py.into()))
            }
        }
    }
}


