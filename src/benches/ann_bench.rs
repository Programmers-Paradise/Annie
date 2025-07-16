// benches/ann_bench.rs

use criterion::{criterion_group, criterion_main, Criterion};
use rust_annie::index::{AnnIndex, Distance};
use rust_annie::hnsw::HnswIndex;
use rand::Rng;

fn generate_data(n: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| (0..dim).map(|_| rng.gen()).collect()).collect()
}

fn bench_brute_vs_hnsw(c: &mut Criterion) {
    let dimensions = [16, 64, 128];
    let num_points = 1000;
    let k = 10;

    for &dim in &dimensions {
        let data = generate_data(num_points, dim);
        let query = data[0].clone();
        let ids: Vec<i64> = (0..num_points).map(|i| i as i64).collect();

        // Brute force
        let mut ann = AnnIndex::new(dim, Distance::L2).unwrap();
        ann.add_batch(&data, &ids).unwrap();

        c.bench_function(&format!("brute_dim_{}", dim), |b| {
            b.iter(|| {
                let _ = ann.search(&query, k);
            });
        });

        // HNSW
        let mut hnsw = HnswIndex::new(dim, Distance::L2).unwrap();
        hnsw.add_batch(&data, &ids).unwrap();

        c.bench_function(&format!("hnsw_dim_{}", dim), |b| {
            b.iter(|| {
                let _ = hnsw.search(&query, k);
            });
        });
    }
}

criterion_group!(benches, bench_brute_vs_hnsw);
criterion_main!(benches);
