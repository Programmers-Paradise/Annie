# GPU Performance Optimization Guide

## Memory Management
- Use `GpuMemoryPool` for buffer reuse
- Monitor usage with `memory_usage()`
- Pre-allocate buffers during initialization

## Multi-GPU Setup
```rust
// Distribute across 4 GPUs
for device_id in 0..4 {
    set_active_device(device_id)?;
    // Add portion of dataset
}
```

## Precision Selection
```rust
gpu_backend.set_precision(Precision::Fp16);  // 2x memory savings
```

## Kernel Selection
We provide optimized kernels for:
- `l2_distance_fp32.ptx`
- `l2_distance_fp16.ptx`
- `l2_distance_int8.ptx`

## Benchmark Results

Command: `cargo bench --features cuda`

Typical results on V100:
- FP32: 15ms @ 1M vectors
- FP16: 9ms @ 1M vectors
- INT8: 6ms @ 1M vectors (with quantization)