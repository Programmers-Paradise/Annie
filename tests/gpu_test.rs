#[cfg(feature = "gpu")]
#[test]
fn test_l2_gpu() {
    use annie::gpu::gpu::l2_distance_gpu;

    let dim = 3;
    let n_queries = 1;
    let n_vectors = 1;
    let queries = vec![1.0, 2.0, 3.0];
    let corpus = vec![1.0, 0.0, 0.0];

    let out = l2_distance_gpu(&queries, &corpus, dim, n_queries, n_vectors)
        .expect("Failed to compute L2 distance on GPU");
    assert!((out[0] - 13.0).abs() < 1e-3); // (2² + 3²) = 13
}
