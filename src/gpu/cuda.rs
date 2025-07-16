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
        // Set active device
        cust::device::set_device(device_id as u32).map_err(GpuError::Cuda)?;
        
        // Create CUDA context
        let ctx = cust::quick_init().map_err(GpuError::Cuda)?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).map_err(GpuError::Cuda)?;
        
        // Get kernel based on precision
        let kernel_name = format!("l2_distance_{}", precision.kernel_suffix());
        let ptx = match precision {
            Precision::Fp32 => include_str!("kernels/l2_kernel_fp32.ptx"),
            Precision::Fp16 => include_str!("kernels/l2_kernel_fp16.ptx"),
            Precision::Int8 => include_str!("kernels/l2_kernel_int8.ptx"),
        };
        
        // Load module
        let module = Module::from_ptx(ptx, &[]).map_err(GpuError::Cuda)?;
        let func = module.get_function(&kernel_name).map_err(GpuError::Cuda)?;
        
        // Convert data to target precision
        let (queries_conv, corpus_conv) = convert_data(queries, corpus, precision)?;
        
        // Allocate buffers using memory pool
        let mut pool = MEMORY_POOL.lock().unwrap();
        let query_buf = pool.get_buffer(device_id, queries_conv.len(), precision);
        let corpus_buf = pool.get_buffer(device_id, corpus_conv.len(), precision);
        let mut out_buf = pool.get_buffer(device_id, n_queries * n_vectors, Precision::Fp32);
        
        // Copy data to device
        let d_query = DeviceBuffer::from_slice(&queries_conv).map_err(GpuError::Cuda)?;
        let d_corpus = DeviceBuffer::from_slice(&corpus_conv).map_err(GpuError::Cuda)?;
        let mut d_out: DeviceBuffer<f32> = unsafe {
            DeviceBuffer::from_raw_parts(
                out_buf.as_mut_ptr() as *mut f32,
                out_buf.len() / std::mem::size_of::<f32>()
            )
        };
        
        // Launch kernel
        let block_size = 256;
        let grid_size = (n_queries as u32 + block_size - 1) / block_size;
        
        unsafe {
            launch!(
                func<<<grid_size, block_size, 0, stream>>>(
                    d_query.as_device_ptr(),
                    d_corpus.as_device_ptr(),
                    d_out.as_device_ptr(),
                    n_queries as i32,
                    n_vectors as i32,
                    dim as i32
                )
            ).map_err(GpuError::Cuda)?;
        }
        
        // Synchronize and copy results back
        stream.synchronize().map_err(GpuError::Cuda)?;
        let mut results = vec![0.0f32; n_queries * n_vectors];
        d_out.copy_to(&mut results).map_err(GpuError::Cuda)?;
        
        // Return buffers to pool
        pool.return_buffer(device_id, query_buf, queries_conv.len(), precision);
        pool.return_buffer(device_id, corpus_buf, corpus_conv.len(), precision);
        pool.return_buffer(device_id, out_buf, results.len(), Precision::Fp32);
        
        Ok(results)
    }

    fn memory_usage(device_id: usize) -> Result<(usize, usize), GpuError> {
        MEMORY_POOL.lock().unwrap().memory_usage(device_id)
            .ok_or_else(|| GpuError::Allocation("Device not initialized".into()))
    }

    fn device_count() -> usize {
        cust::device::get_count().unwrap_or(0) as usize
    }
}

/// Convert data to target precision
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
            unsafe { 
                std::slice::from_raw_parts(
                    data.as_ptr() as *const f32, 
                    data.len() / std::mem::size_of::<f32>()
                ) 
            },
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