//! Feed arbitrary vectors, IDs and queries to both index types. Any panic is a
//! bug: invalid input must surface as an error, never a crash.
#![no_main]

use libfuzzer_sys::fuzz_target;
use ndarray::Array2;
use rust_annie::filters::Filter;
use rust_annie::hnsw_index::{HnswConfig, HnswIndex};
use rust_annie::index::AnnIndex;
use rust_annie::metrics::Distance;

const METRICS: [fn() -> Distance; 6] = [
    Distance::Euclidean,
    Distance::Cosine,
    Distance::Manhattan,
    Distance::Chebyshev,
    Distance::Angular,
    Distance::Canberra,
];

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    let dim = (data[0] % 8) as usize + 1;
    let k = (data[1] % 16) as usize;
    let metric = METRICS[data[2] as usize % METRICS.len()]();
    let floats: Vec<f32> = data[4..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if floats.len() < dim {
        return;
    }
    let n = floats.len() / dim - 1;
    let query = &floats[..dim];
    let rows = Array2::from_shape_vec((n, dim), floats[dim..dim * (n + 1)].to_vec()).unwrap();
    let ids: Vec<i64> = (0..n as i64)
        .map(|i| i % (data[3] as i64 + 1) - 1)
        .collect();

    let mut brute = AnnIndex::create(dim, metric.clone()).unwrap();
    let _ = brute.add_rows(rows.view(), &ids);
    let _ = brute.search_vec(query, k, None);
    let _ = brute.search_vec(query, k, Some(&Filter::id_range(0, 3)));
    let _ = brute.search_batch_vec(rows.view(), k, None);
    let _ = brute.remove_ids(&ids[..ids.len().min(2)]);

    if let Ok(mut hnsw) = HnswIndex::new(dim, metric, HnswConfig::default()) {
        let _ = hnsw.add_rows(rows.view(), &ids);
        let _ = hnsw.search_vec(query, k, None, None);
        let _ = hnsw.remove_ids(&ids[..ids.len().min(2)]);
        let _ = hnsw.search_vec(query, k, Some(k * 2), None);
    }
});
