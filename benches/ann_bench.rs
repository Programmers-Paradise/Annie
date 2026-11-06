//! Criterion micro-benchmarks for the Rust core.
//!
//! Run with `cargo bench`. For end-to-end comparisons against other
//! libraries, see `benchmarks/compare.py`.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;
use rust_annie::hnsw_index::{HnswConfig, HnswIndex};
use rust_annie::index::AnnIndex;
use rust_annie::metrics::Distance;

/// Deterministic pseudo-random data (xorshift), avoiding a `rand` dependency.
fn random_data(n: usize, dim: usize, seed: u64) -> Array2<f32> {
    let mut s = seed.max(1);
    Array2::from_shape_simple_fn((n, dim), || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 40) as f32 / (1u64 << 24) as f32
    })
}

fn bench_search(c: &mut Criterion) {
    let n = 10_000;
    let k = 10;
    let mut group = c.benchmark_group("search_k10_n10000");
    for &dim in &[16usize, 128] {
        let data = random_data(n, dim, 42);
        let ids: Vec<i64> = (0..n as i64).collect();
        let queries = random_data(64, dim, 7);

        let mut brute = AnnIndex::create(dim, Distance::Euclidean()).unwrap();
        brute.add_rows(data.view(), &ids).unwrap();
        group.bench_with_input(BenchmarkId::new("brute_single", dim), &dim, |b, _| {
            b.iter(|| {
                brute
                    .search_vec(queries.row(0).as_slice().unwrap(), k, None)
                    .unwrap()
            })
        });
        group.bench_with_input(BenchmarkId::new("brute_batch64", dim), &dim, |b, _| {
            b.iter(|| brute.search_batch_vec(queries.view(), k, None).unwrap())
        });

        let mut hnsw = HnswIndex::new(dim, Distance::Euclidean(), HnswConfig::default()).unwrap();
        hnsw.add_rows(data.view(), &ids).unwrap();
        group.bench_with_input(BenchmarkId::new("hnsw_single", dim), &dim, |b, _| {
            b.iter(|| {
                hnsw.search_vec(queries.row(0).as_slice().unwrap(), k, None, None)
                    .unwrap()
            })
        });
    }
    group.finish();
}

fn bench_build(c: &mut Criterion) {
    let data = random_data(5_000, 64, 1);
    let ids: Vec<i64> = (0..5_000).collect();
    let mut group = c.benchmark_group("build_n5000_d64");
    group.sample_size(10);
    group.bench_function("hnsw", |b| {
        b.iter(|| {
            let mut h = HnswIndex::new(64, Distance::Euclidean(), HnswConfig::default()).unwrap();
            h.add_rows(data.view(), &ids).unwrap();
        })
    });
    group.finish();
}

criterion_group!(benches, bench_search, bench_build);
criterion_main!(benches);
