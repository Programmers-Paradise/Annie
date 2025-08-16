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

## Backend Creation and Error Handling

When creating a GPU backend, the process now returns a `Result` type, which can fail with a specific error. This allows for more robust error handling when initializing the backend.

### Example
```rust
use crate::errors::BackendCreationError;

fn create_backend() -> Result<BackendEnum, BackendCreationError> {
    BackendEnum::new("gpu", 128, Distance::Euclidean)
}

match create_backend() {
    Ok(backend) => {
        // Proceed with using the backend
    }
    Err(e) => {
        eprintln!("Error creating backend: {}", e);
    }
}
```

### Possible Errors
- `UnsupportedBackend`: This error occurs if an unsupported backend type is specified. The available backends are 'brute', 'hnsw', and 'gpu'.

## GPU Error Conditions

When using GPU features, be aware of potential error conditions that can arise:

- **Invalid Input Sizes**: Ensure that input vectors are not empty and have matching dimensions.
- **Mismatched Dimensions**: Inputs to GPU functions must have compatible dimensions.
- **Large Allocations**: Attempting to allocate excessively large vectors may result in out-of-memory (OOM) errors.

### Example
```rust
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
```

## HIP Kernel Compilation

For systems using the ROCm platform, HIP kernel compilation is handled with enhanced environment validation and error handling. Ensure that `hipcc` is available in your `PATH` and that the kernel source file exists.

### Example
```rust
#[cfg(feature = "rocm")]
fn compile_hip_kernel() {
    use std::env;
    use std::path::PathBuf;
    // Validate and sanitize build environment
    let hipcc_path = match which::which("hipcc") {
        Ok(path) => path,
        Err(_) => {
            eprintln!("hipcc not found in PATH. Aborting HIP kernel build.");
            return;
        }
    };
    let kernel_src = PathBuf::from("kernels/l2_kernel.hip");
    let kernel_out = PathBuf::from("kernels/l2_kernel.hsaco");
    if !kernel_src.exists() {
        eprintln!("Kernel source {:?} does not exist.", kernel_src);
        return;
    }
    // Only allow known safe arguments
    let args = ["--genco", "-o", kernel_out.to_str().unwrap(), kernel_src.to_str().unwrap()];
    let status = Command::new(hipcc_path)
        .args(&args)
        .env_clear()
        .envs(env::vars().filter(|(k,_)| k == "PATH" || k == "HOME"))
        .status();
    let status = match status {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to execute hipcc: {}", e);
            return;
        }
    };
    
    if !status.success() {
        eprintln!("HIP kernel compilation failed with exit code: {:?}", status.code());
        return;
    }
    
    println!("cargo:rerun-if-changed=kernels/l2_kernel.hip");
}
```

## Benchmark Results

Command: `cargo bench --features cuda`

Typical results on V100:
- FP32: 15ms @ 1M vectors
- FP16: 9ms @ 1M vectors
- INT8: 6ms @ 1M vectors (with quantization)

This updated documentation reflects the changes in the GPU backend creation process, highlighting the new error handling mechanism and providing guidance on how to handle potential errors effectively. Additionally, it includes information on GPU error conditions and HIP kernel compilation to help users avoid common pitfalls when using GPU features.