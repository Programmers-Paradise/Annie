# GPU Memory Management Guide

This document describes the GPU memory management improvements in Annie to prevent memory leaks and handle memory pressure effectively.

## Overview

Annie's GPU memory management system now uses RAII (Resource Acquisition Is Initialization) patterns to ensure automatic cleanup of GPU resources, preventing memory leaks and improving system stability.

## Key Features

### RAII Memory Management

The new system provides automatic cleanup through Rust's Drop trait:

```rust
use annie::gpu::memory::GpuMemoryPool;
use annie::gpu::Precision;

let pool = GpuMemoryPool::new();

// Managed buffer automatically returns to pool when dropped
{
    let buffer = pool.get_managed_buffer(0, 1024, Precision::Fp32);
    // Use buffer...
} // Buffer automatically returned to pool here
```

### Memory Pressure Handling

The system monitors memory pressure and automatically cleans up fragmented buffers:

```rust
// Check memory pressure (0.0 = no pressure, 1.0 = at limit)
let pressure = pool.memory_pressure(device_id);
if let Some(p) = pressure {
    if p > 0.8 {
        println!("High memory pressure: {:.1}%", p * 100.0);
    }
}

// Emergency cleanup if needed
if pressure.unwrap_or(0.0) > 0.9 {
    pool.emergency_cleanup(device_id);
}
```

### Batch Operations

For multiple buffers, use batch operations for efficient cleanup:

```rust
let mut batch = pool.create_buffer_batch(device_id);

// Add multiple buffers
batch.add_buffer(vec![0u8; 256], 64, Precision::Fp32);
batch.add_buffer(vec![0u8; 128], 64, Precision::Fp16);

// All buffers automatically cleaned up when batch is dropped
```

## Memory Pool Configuration

### Setting Pool Limits

```rust
// Set maximum pool size (1GB)
pool.set_max_pool_size(device_id, 1024 * 1024 * 1024);

// Monitor total memory usage across all devices
let (total_allocated, total_peak) = pool.total_memory_usage();
println!("Total allocated: {} bytes, Peak: {} bytes", total_allocated, total_peak);
```

### Memory Usage Monitoring

```rust
// Get memory usage for a specific device
if let Some((allocated, peak)) = pool.memory_usage(device_id) {
    println!("Device {}: {} bytes allocated, {} bytes peak", 
             device_id, allocated, peak);
}
```

## Error Handling Improvements

### Automatic Cleanup on Errors

The improved CUDA backend now automatically cleans up resources even when errors occur:

```rust
// All resources are automatically cleaned up even if this fails
let result = l2_distance_gpu(&queries, &corpus, dim, n_queries, n_vectors, 
                            device_id, precision);
match result {
    Ok(distances) => {
        // Process results
    }
    Err(e) => {
        // No manual cleanup needed - RAII handles it
        eprintln!("GPU operation failed: {}", e);
    }
}
```

### Enhanced Error Types

New error types provide better diagnostics:

```rust
use annie::gpu::GpuError;

match gpu_operation() {
    Err(GpuError::InvalidInput(msg)) => {
        eprintln!("Invalid input: {}", msg);
    }
    Err(GpuError::Allocation(msg)) => {
        eprintln!("Memory allocation failed: {}", msg);
        // Consider emergency cleanup
        pool.emergency_cleanup_all();
    }
    Err(e) => eprintln!("Other GPU error: {}", e),
    Ok(_) => {}
}
```

## Best Practices

### 1. Use Managed Buffers

Always prefer managed buffers over raw buffer operations:

```rust
// Preferred: Automatic cleanup
let buffer = pool.get_managed_buffer(device_id, size, precision);

// Avoid: Manual cleanup required
let raw_buffer = pool.get_buffer(device_id, size, precision);
// ... use buffer ...
pool.return_buffer(device_id, raw_buffer, size, precision); // Easy to forget!
```

### 2. Monitor Memory Pressure

Regularly check memory pressure in long-running applications:

```rust
fn check_memory_health(pool: &GpuMemoryPool, device_id: usize) {
    if let Some(pressure) = pool.memory_pressure(device_id) {
        if pressure > 0.7 {
            log::warn!("High memory pressure on device {}: {:.1}%", 
                      device_id, pressure * 100.0);
        }
        if pressure > 0.9 {
            log::error!("Critical memory pressure, performing emergency cleanup");
            pool.emergency_cleanup(device_id);
        }
    }
}
```

### 3. Set Appropriate Pool Limits

Configure pool limits based on your GPU memory:

```rust
fn configure_gpu_memory(pool: &GpuMemoryPool) {
    // Get GPU memory info and set conservative limits
    if let Ok(stats) = get_memory_stats(0) {
        let max_pool = stats.total * 70 / 100; // Use 70% of total memory
        pool.set_max_pool_size(0, max_pool);
    }
}
```

### 4. Batch Related Operations

Group related GPU operations to minimize memory fragmentation:

```rust
// Group operations that use similar buffer sizes
let mut batch = pool.create_buffer_batch(device_id);
for data_chunk in data_chunks {
    batch.add_buffer(process_chunk(data_chunk), size, precision);
}
// All buffers cleaned up together
```

## Performance Considerations

### Memory Reuse

The pool automatically reuses buffers of the same size and precision:

- Reduces allocation overhead
- Minimizes memory fragmentation
- Improves GPU kernel launch latency

### Fragmentation Management

The system automatically manages fragmentation:

- Monitors buffer type diversity
- Cleans up underutilized buffer types
- Maintains optimal cache hit rates

### Memory Pressure Response

Under memory pressure, the system:

1. Reduces cached buffer counts
2. Removes fragmented buffer types
3. Provides emergency cleanup options
4. Reports pressure metrics for monitoring

## Troubleshooting

### High Memory Usage

If you see consistently high memory usage:

1. Check for long-lived buffer references
2. Monitor memory pressure regularly
3. Consider more frequent emergency cleanup
4. Reduce pool size limits

### Performance Issues

If GPU performance degrades:

1. Monitor fragmentation levels
2. Use batch operations for related work
3. Avoid very small or very large buffer requests
4. Consider warmup operations for consistent performance

### Error Recovery

For robust applications, implement error recovery:

```rust
fn robust_gpu_operation(pool: &GpuMemoryPool, data: &[f32]) -> Result<Vec<f32>, GpuError> {
    match gpu_operation(data) {
        Ok(result) => Ok(result),
        Err(GpuError::Allocation(_)) => {
            // Try cleanup and retry once
            pool.emergency_cleanup_all();
            gpu_operation(data)
        }
        Err(e) => Err(e),
    }
}
```

## Migration Guide

### From Old Memory Management

If upgrading from the previous memory management system:

1. Replace manual buffer management with managed buffers
2. Remove manual cleanup code (now automatic)
3. Add memory pressure monitoring
4. Update error handling to use new error types

### Backward Compatibility

The old memory pool interface remains available but is deprecated:

- `get_buffer()` and `return_buffer()` still work
- New code should use `get_managed_buffer()`
- Existing code will continue to function

## Summary

The new GPU memory management system provides:

- ✅ Automatic resource cleanup via RAII
- ✅ Memory leak prevention
- ✅ Memory pressure monitoring
- ✅ Fragmentation management
- ✅ Enhanced error handling
- ✅ Performance optimization
- ✅ Backward compatibility

This ensures robust, efficient GPU memory usage while maintaining ease of use and preventing common memory management pitfalls.