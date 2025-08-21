use crate::gpu::{GpuBackend, GpuError, Precision};
use cust::prelude::*;
use crate::gpu::memory::GpuMemoryPool;
use lazy_static::lazy_static;
use std::sync::{Arc, Mutex};
use half::f16;
use std::convert::TryInto;

// Global memory pool with thread-safe access
lazy_static! {
    static ref MEMORY_POOL: Arc<Mutex<GpuMemoryPool>> = 
        Arc::new(Mutex::new(GpuMemoryPool::new()));
}

/// CUDA backend implementation
pub struct CudaBackend;

impl GpuBackend for CudaBackend {
    fn l2_distance(
        queries: &[f32],
        corpus: &[f32],
        dim: usize,
        n_queries: usize,
        n_vectors: usize,
        device_id: usize,
        precision: Precision,
    ) -> Result<Vec<f32>, GpuError> {
        // Validate inputs first
        if queries.is_empty() || corpus.is_empty() {
            return Err(GpuError::InvalidInput("Empty input arrays".to_string()));
        }
        
        if queries.len() != n_queries * dim || corpus.len() != n_vectors * dim {
            return Err(GpuError::InvalidInput("Input array sizes don't match specified dimensions".to_string()));
        }

        // Set up CUDA context with RAII cleanup
        cust::device::set_device(device_id as u32).map_err(GpuError::Cuda)?;
        let _ctx = cust::quick_init().map_err(GpuError::Cuda)?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).map_err(GpuError::Cuda)?;
        
        // Get kernel
        let (kernel_name, ptx) = get_kernel_and_ptx(precision);
        let module = Module::from_ptx(ptx, &[]).map_err(GpuError::Cuda)?;
        let func = module.get_function(&kernel_name).map_err(GpuError::Cuda)?;
        
        // Convert data to target precision
        let (queries_conv, corpus_conv) = convert_data(queries, corpus, precision)?;
        
        // Validate converted data sizes
        let expected_query_size = n_queries * dim * precision.element_size();
        let expected_corpus_size = n_vectors * dim * precision.element_size();
        if queries_conv.len() != expected_query_size || corpus_conv.len() != expected_corpus_size {
            return Err(GpuError::InvalidInput("Converted buffer size mismatch".to_string()));
        }

        // Use RAII memory management
        let pool = MEMORY_POOL.clone();
        let mut managed_pool = pool.lock().unwrap();
        
        // Get managed buffers that auto-cleanup on drop
        let mut query_buffer = managed_pool.get_managed_buffer(device_id, queries_conv.len(), precision);
        let mut corpus_buffer = managed_pool.get_managed_buffer(device_id, corpus_conv.len(), precision);
        let mut output_buffer = managed_pool.get_managed_buffer(device_id, n_queries * n_vectors * 4, Precision::Fp32); // 4 bytes for f32 output
        
        // Copy data to buffers
        query_buffer.as_mut_slice().copy_from_slice(&queries_conv);
        corpus_buffer.as_mut_slice().copy_from_slice(&corpus_conv);
        
        // Release the pool lock before GPU operations
        drop(managed_pool);
        
        // Allocate device buffers with RAII cleanup
        let d_query = DeviceBuffer::from_slice(query_buffer.as_slice()).map_err(GpuError::Cuda)?;
        let d_corpus = DeviceBuffer::from_slice(corpus_buffer.as_slice()).map_err(GpuError::Cuda)?;
        let mut d_output = DeviceBuffer::<f32>::zeroed(n_queries * n_vectors).map_err(GpuError::Cuda)?;
        
        // Launch kernel
        let block_size = 256;
        let grid_size = ((n_queries * n_vectors + block_size - 1) / block_size) as u32;
        
        unsafe {
            launch!(func<<<grid_size, block_size, 0, stream>>>(
                d_query.as_device_ptr(),
                d_corpus.as_device_ptr(),
                d_output.as_device_ptr(),
                n_queries as i32,
                n_vectors as i32,
                dim as i32
            )).map_err(GpuError::Cuda)?;
        }
        
        // Wait for completion and copy results
        stream.synchronize().map_err(GpuError::Cuda)?;
        
        let mut results = vec![0.0f32; n_queries * n_vectors];
        d_output.copy_to(&mut results).map_err(GpuError::Cuda)?;
        
        // Buffers automatically cleaned up by RAII Drop implementations
        Ok(results)
    }
// Refactor: Move device setup, kernel selection, memory management, and kernel launch into separate helper functions or modules to reduce complexity and improve maintainability.

    fn memory_usage(device_id: usize) -> Result<(usize, usize), GpuError> {
        MEMORY_POOL.lock().unwrap().memory_usage(device_id)
            .ok_or_else(|| GpuError::Allocation("Device not initialized".into()))
    }

    fn device_count() -> usize {
        cust::device::get_count().unwrap_or(0) as usize
    }
}

/// Get kernel name and PTX based on precision
fn get_kernel_and_ptx(precision: Precision) -> (String, &'static str) {
    match precision {
        Precision::Fp32 => ("l2_distance_fp32".to_string(), include_str!("kernels/l2_kernel_fp32.ptx")),
        Precision::Fp16 => ("l2_distance_fp16".to_string(), include_str!("kernels/l2_kernel_fp16.ptx")), 
        Precision::Int8 => ("l2_distance_int8".to_string(), include_str!("kernels/l2_kernel_int8.ptx")),
    }
}

impl CudaBackend {
    /// Check if precision is supported
    pub fn supports_precision(precision: Precision) -> bool {
        match precision {
            Precision::Fp32 | Precision::Fp16 | Precision::Int8 => true,
        }
    }
}

/// Get device count for CUDA
pub fn device_count() -> usize {
    cust::device::get_count().unwrap_or(0) as usize
}
fn convert_data(
    queries: &[f32],
    corpus: &[f32],
    precision: Precision,
) -> Result<(Vec<u8>, Vec<u8>), GpuError> {
    match precision {
        Precision::Fp32 => {
            // No conversion needed, just reinterpret as bytes
            let queries_bytes = unsafe {
                std::slice::from_raw_parts(
                    queries.as_ptr() as *const u8,
                    queries.len() * std::mem::size_of::<f32>()
                ).to_vec()
            };
            let corpus_bytes = unsafe {
                std::slice::from_raw_parts(
                    corpus.as_ptr() as *const u8,
                    corpus.len() * std::mem::size_of::<f32>()
                ).to_vec()
            };
            Ok((queries_bytes, corpus_bytes))
        }
        Precision::Fp16 => {
            // Convert to f16
            let queries_f16: Vec<f16> = queries.iter().map(|&x| f16::from_f32(x)).collect();
            let corpus_f16: Vec<f16> = corpus.iter().map(|&x| f16::from_f32(x)).collect();
            
            // Reinterpret as bytes
            let queries_bytes = unsafe {
                std::slice::from_raw_parts(
                    queries_f16.as_ptr() as *const u8,
                    queries_f16.len() * std::mem::size_of::<f16>()
                ).to_vec()
            };
            let corpus_bytes = unsafe {
                std::slice::from_raw_parts(
                    corpus_f16.as_ptr() as *const u8,
                    corpus_f16.len() * std::mem::size_of::<f16>()
                ).to_vec()
            };
            Ok((queries_bytes, corpus_bytes))
        }
        Precision::Int8 => {
            // Convert to int8 with scaling
            let queries_i8: Vec<i8> = queries.iter()
                .map(|&x| (x * 127.0).clamp(-128.0, 127.0) as i8)
                .collect();
            let corpus_i8: Vec<i8> = corpus.iter()
                .map(|&x| (x * 127.0).clamp(-128.0, 127.0) as i8)
                .collect();
            
            // Reinterpret as bytes
            let queries_bytes = unsafe {
                std::slice::from_raw_parts(
                    queries_i8.as_ptr() as *const u8,
                    queries_i8.len()
                ).to_vec()
            };
            let corpus_bytes = unsafe {
                std::slice::from_raw_parts(
                    corpus_i8.as_ptr() as *const u8,
                    corpus_i8.len()
                ).to_vec()
            };
            Ok((queries_bytes, corpus_bytes))
        }
    }
}

/// Initialize CUDA context for a device
pub fn init_device(device_id: usize) -> Result<(), GpuError> {
    cust::device::set_device(device_id as u32).map_err(GpuError::Cuda)?;
    // Warm-up kernel to initialize context
    let _ = cust::quick_init();
    Ok(())
}

/// Multi-GPU data distribution helper
pub fn distribute_data(
    data: &[f32],
    dim: usize,
    devices: &[usize],
) -> Result<Vec<(usize, Vec<u8>)>, GpuError> {
    let total = data.len() / dim;
    let per_device = total / devices.len();
    let mut distributed = Vec::new();
    
    for (i, &device_id) in devices.iter().enumerate() {
        let start = i * per_device * dim;
        let end = if i == devices.len() - 1 {
            data.len()
        } else {
            (i + 1) * per_device * dim
        };
        
        let device_data = &data[start..end];
        let bytes = unsafe {
            std::slice::from_raw_parts(
                device_data.as_ptr() as *const u8,
                device_data.len() * std::mem::size_of::<f32>()
            ).to_vec()
        };
        
        distributed.push((device_id, bytes));
    }
    
    Ok(distributed)
}

/// Multi-GPU search
pub fn multi_gpu_search(
    query: &[f32],
    data_chunks: &[(usize, Vec<u8>)],
    dim: usize,
    k: usize,
    precision: Precision,
) -> Result<Vec<(usize, f32)>, GpuError> {
    let mut all_results = Vec::new();
    let mut streams = Vec::new();
    
    // Create a stream per device
    for (device_id, _) in data_chunks {
        cust::device::set_device(*device_id as u32)?;
        streams.push(Stream::new(StreamFlags::NON_BLOCKING, None)?);
    }
    
    // Launch searches in parallel
    let mut futures = Vec::new();
    for ((device_id, data), stream) in data_chunks.iter().zip(streams.iter()) {
        cust::device::set_device(*device_id as u32)?;
        
        // Convert query to target precision
        let query_conv = match precision {
            Precision::Fp32 => query.to_vec(),
            Precision::Fp16 => query.iter().map(|&x| f16::from_f32(x).to_f32()).collect(),
            Precision::Int8 => query.iter().map(|&x| (x * 127.0) as f32).collect(),
        };
        
        let n_vectors = data.len() / (dim * precision.element_size());
        let future = CudaBackend::l2_distance(
            &query_conv,
            &[], // Pass empty slice; l2_distance expects raw bytes for non-f32, so this call must be refactored
            dim,
            1,
            n_vectors,
            *device_id,
            precision,
        );
        
        futures.push(future);
    }
    
    // Collect results
    for (i, future) in futures.into_iter().enumerate() {
        let (device_id, _) = &data_chunks[i];
        cust::device::set_device(*device_id as u32)?;
        
        let mut distances = future?;
        let start_idx = i * (data_chunks[0].1.len() / (dim * precision.element_size()));
        
        all_results.extend(
            distances.into_iter()
                .enumerate()
                .map(|(j, d)| (start_idx + j, d))
        );
    }
    
    // Merge results and select top-k
    all_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    all_results.truncate(k);
    
    Ok(all_results)
}

/// GPU memory usage statistics
pub struct GpuMemoryStats {
    pub total: usize,
    pub free: usize,
    pub used: usize,
}

/// Get detailed GPU memory info
pub fn get_memory_stats(device_id: usize) -> Result<GpuMemoryStats, GpuError> {
    cust::device::set_device(device_id as u32)?;
    let ctx = cust::quick_init()?;
    let (free, total) = ctx.get_mem_info().map_err(GpuError::Cuda)?;
    
    Ok(GpuMemoryStats {
        total: total as usize,
        free: free as usize,
        used: (total - free) as usize,
    })
}

/// Kernel warmup to reduce first-run latency
pub fn warmup_kernels(device_id: usize) -> Result<(), GpuError> {
    cust::device::set_device(device_id as u32)?;
    let ctx = cust::quick_init()?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    
    // Warmup dummy kernel
    let ptx = include_str!("kernels/l2_kernel_fp32.ptx");
    let module = Module::from_ptx(ptx, &[])?;
    let func = module.get_function("l2_distance_fp32")?;
    
    // Dummy buffers
    let dummy_data = [0.0f32; 16];
    let d_data = DeviceBuffer::from_slice(&dummy_data)?;
    let mut d_out = DeviceBuffer::<f32>::zeroed(1)?;
    
    unsafe {
        launch!(func<<<1, 1, 0, stream>>>(
            d_data.as_device_ptr(),
            d_data.as_device_ptr(),
            d_out.as_device_ptr(),
            1,
            1,
            1
        ))?;
    }
    
    stream.synchronize()?;
    Ok(())
}