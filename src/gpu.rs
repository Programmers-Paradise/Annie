#[cfg(feature = "gpu")]
mod gpu {
    use cust::prelude::*;

    pub fn l2_distance_gpu(queries: &[f32], corpus: &[f32], dim: usize, n_queries: usize, n_vectors: usize) -> Result<Vec<f32>, cust::error::CudaError> {
        // Load PTX and create context
        let ptx = include_str!("kernels/l2_kernel.ptx");
        let _ctx = cust::quick_init()?;
        let module = Module::from_ptx(ptx, &[])?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        // Allocate device buffers
        let query_buf = DeviceBuffer::from_slice(queries)?;
        let corpus_buf = DeviceBuffer::from_slice(corpus)?;
        let mut out_buf = DeviceBuffer::zeroed(n_queries * n_vectors)?;

        // Launch kernel
        let func = module.get_function("l2_distance_kernel")?;
        unsafe {
            launch!(func<<<n_queries as u32, n_vectors as u32, 0, stream>>>(
                query_buf.as_device_ptr(),
                corpus_buf.as_device_ptr(),
                out_buf.as_device_ptr(),
                n_queries as i32,
                n_vectors as i32,
                dim as i32
            ))?;
        }

        stream.synchronize()?;

        let mut out = vec![0.0f32; n_queries * n_vectors];
        out_buf.copy_to(&mut out)?;
        out
    }
}
