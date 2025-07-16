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
        set_active_device(device_id)?;
        let ctx = create_cuda_context()?;
        let stream = create_stream()?;
        let (kernel_name, ptx) = get_kernel_and_ptx(precision);
        let module = load_module(ptx)?;
        let func = get_kernel_function(&module, &kernel_name)?;
        let (queries_conv, corpus_conv) = convert_data(queries, corpus, precision)?;
        let mut pool = MEMORY_POOL.lock().unwrap();
        if queries_conv.len() != n_queries * dim * precision.element_size() || corpus_conv.len() != n_vectors * dim * precision.element_size() {
            return Err(GpuError::InvalidInput("Input buffer size does not match expected dimensions".to_string()));
        }
        let (query_buf, corpus_buf, mut out_buf) = allocate_buffers(&mut pool, device_id, &queries_conv, &corpus_conv, n_queries, n_vectors, precision);
        let (d_query, d_corpus, mut d_out) = copy_data_to_device(&queries_conv, &corpus_conv, &mut out_buf)?;
        launch_l2_kernel(&func, &stream, &d_query, &d_corpus, &mut d_out, n_queries, n_vectors, dim)?;
        let results = copy_results_from_device(&stream, &mut d_out, n_queries, n_vectors)?;
        Ok(results)
        // return_buffers must be called after results are no longer needed, or ensure device memory is not dropped before host copy completes
        return_buffers(&mut pool, device_id, query_buf, corpus_buf, out_buf, &queries_conv, &corpus_conv, &results, precision);
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