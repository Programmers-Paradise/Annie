#[macro_use]
extern crate criterion;
use criterion::Criterion;
use rust_annie::backends::BackendEnum;
use rust_annie::metrics::Distance;
use rand::prelude::*;

fn bench_gpu_search(c: &mut Criterion) {
    let mut backend = BackendEnum::new("gpu", 128, Distance::Euclidean);
        .expect("Failed to create GPU backend for benchmark");
    
    // Add 1M random vectors
    let mut rng = rand::thread_rng();
    if let Some(batch_add) = backend.batch_add_method() {
        let vectors: Vec<Vec<f32>> = (0..1_000_000)
            .map(|_| (0..128).map(|_| rng.gen()).collect())
            .collect();
        batch_add(vectors);
    } else {
        for _ in 0..1_000_000 {
            let vec: Vec<f32> = (0..128).map(|_| rng.gen()).collect();
            backend.add(vec);
        }
    }
    
    let query = vec![0.5; 128];
    
    c.bench_function("GPU ANN Search", |b| {
        b.iter(|| backend.search(&query, 10))
    });
    
    // Precision benchmarks
    if let BackendEnum::Gpu(gpu) = &mut backend {
        gpu.set_precision(Precision::Fp16);
        c.bench_function("GPU ANN Search (FP16)", |b| {
            b.iter(|| gpu.search(&query, 10))
        });
    }
}

criterion_group!(benches, bench_gpu_search);
criterion_main!(benches);