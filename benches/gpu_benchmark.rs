#[macro_use]
extern crate criterion;
use criterion::Criterion;
use rust_annie::backends::BackendEnum;
use rust_annie::metrics::Distance;
// Import the AnnBackend trait to use its methods like `add` and `search`.
use rust_annie::backends::ann_backend::AnnBackend;
use rust_annie::gpu::Precision;
use rand::prelude::*;

fn bench_gpu_search(c: &mut Criterion) {
    // FIX: Handle the Result from `new`. 
    // .expect() is used here because a failure indicates a benchmark setup error, which should cause a panic.
    let mut backend = BackendEnum::new("gpu", 128, Distance::Euclidean)
        .expect("Failed to create GPU backend for benchmark. Is the backend name correct?");
    
    // Add 1M random vectors
    let mut rng = rand::thread_rng();

    // FIX: The original code called a non-existent `batch_add_method`.
    // The correct approach is to call the `add` method from the AnnBackend trait in a loop.
    for _ in 0..1_000_000 {
        let vec: Vec<f32> = (0..128).map(|_| rng.gen()).collect();
        backend.add(vec);
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