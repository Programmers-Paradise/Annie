#[cfg(feature = "cuda")]
#[test]
fn test_gpu_error_conditions() {
    use rust_annie::gpu::l2_distance_gpu;
    use rust_annie::gpu::Precision;
    // Invalid input sizes
    let result = l2_distance_gpu(&[], &[], 3, 1, 1, 0, Precision::Fp32);
    assert!(result.is_err(), "Should error on empty input");
    // Mismatched dimensions
    let result = l2_distance_gpu(&[1.0, 2.0], &[1.0, 2.0, 3.0], 3, 1, 1, 0, Precision::Fp32);
    assert!(result.is_err(), "Should error on mismatched input");
    // Large allocation (simulate OOM)
    let big_vec = vec![1.0; 1_000_000_000];
    let result = l2_distance_gpu(&big_vec, &big_vec, 1_000_000_000, 1, 1, 0, Precision::Fp32);
    assert!(result.is_err(), "Should error on huge allocation");
}

#[cfg(feature = "cuda")]
#[test]
fn test_gpu_memory_management_under_errors() {
    use rust_annie::gpu::memory::GpuMemoryPool;
    use rust_annie::gpu::Precision;
    use rust_annie::gpu::l2_distance_gpu;

    let pool = GpuMemoryPool::new();
    let device_id = 0;

    // Get initial memory state
    let initial_usage = pool.memory_usage(device_id).unwrap_or((0, 0));

    // Try operations that should fail, but not leak memory
    let _result1 = l2_distance_gpu(&[], &[], 3, 1, 1, device_id, Precision::Fp32);
    let _result2 = l2_distance_gpu(&[1.0, 2.0], &[1.0, 2.0, 3.0], 3, 1, 1, device_id, Precision::Fp32);

    // Memory should not have increased significantly from failed operations
    let final_usage = pool.memory_usage(device_id).unwrap_or((0, 0));
    
    // There might be some memory allocated for device pool initialization,
    // but it shouldn't grow significantly from failed operations
    assert!(final_usage.0 <= initial_usage.0 + 1024, "Memory leaked during error conditions");
}
