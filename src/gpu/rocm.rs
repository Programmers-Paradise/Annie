use hip_runtime::{hip, memory::DeviceBuffer, stream::Stream};
use crate::gpu::GpuError;

pub struct RocmBackend;

impl super::GpuBackend for RocmBackend {
    fn l2_distance(
        queries: &[f32],
        corpus: &[f32],
        dim: usize,
        n_queries: usize,
        n_vectors: usize,
    ) -> Result<Vec<f32>, GpuError> {
        // Initialize HIP
        hip::init()?;

        // Create HIP kernel (needs to be precompiled)
        let module = hip::Module::load_from_file("kernels/l2_kernel.hsaco")?;
        let kernel = module.get_function("l2_distance_kernel")?;

        // Allocate device memory
        let d_queries = DeviceBuffer::from_slice(queries)?;
        let d_corpus = DeviceBuffer::from_slice(corpus)?;
        let mut d_out = DeviceBuffer::uninitialized(n_queries * n_vectors)?;

        // Set kernel parameters
        let mut args = [
            &d_queries as *const _ as *mut _,
            &d_corpus as *const _ as *mut _,
            &d_out as *const _ as *mut _,
            &(n_queries as i32),
            &(n_vectors as i32),
            &(dim as i32),
        ];

        // Launch kernel
        let grid_size = n_queries as u32;
        let block_size = n_vectors as u32;
        let stream = Stream::new(hip::StreamFlags::NON_BLOCKING, None)?;
        
        unsafe {
            kernel.launch(
                &mut args as *mut _ as *mut *mut _,
                grid_size,
                1,
                1,
                block_size,
                1,
                1,
                0,
                Some(&stream),
            )?;
        }

        // Copy results back
        let mut out = vec![0.0f32; n_queries * n_vectors];
        d_out.copy_to(&mut out)?;
        stream.synchronize()?;

        Ok(out)
    }
}