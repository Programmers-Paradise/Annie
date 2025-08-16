#[cfg(feature = "gpu")]
#[test]
fn test_gpu_error_conditions() {
    use annie::gpu::gpu::l2_distance_gpu;
    // Invalid input sizes
    let result = l2_distance_gpu(&[], &[], 3, 1, 1);
    assert!(result.is_err(), "Should error on empty input");
    // Mismatched dimensions
    let result = l2_distance_gpu(&[1.0, 2.0], &[1.0, 2.0, 3.0], 3, 1, 1);
    assert!(result.is_err(), "Should error on mismatched input");
    // Large allocation (simulate OOM)
    let big_vec = vec![1.0; 1_000_000_000];
    let result = l2_distance_gpu(&big_vec, &big_vec, 1_000_000_000, 1, 1);
    assert!(result.is_err(), "Should error on huge allocation");
}
