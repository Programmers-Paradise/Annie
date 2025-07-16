#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "rocm")]
mod rocm;
mod memory;
mod device;
mod precision;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum GpuError {
    #[error("No compatible GPU backend found")]
    NoBackend,
    #[error("CUDA error: {0}")]
    Cuda(#[from] cust::error::CudaError),
    #[error("ROCm error: {0}")]
    Rocm(#[from] hip_runtime::Status),
    #[error("Memory allocation failed: {0}")]
    Allocation(String),
    #[error("Unsupported precision type")]
    UnsupportedPrecision,
    #[error("Device index out of range: {0}")]
    DeviceIndex(usize),
    #[error("Multi-GPU synchronization error")]
    MultiGpuSync,
}

pub trait GpuBackend {
    fn l2_distance(
        queries: &[f32],
        corpus: &[f32],
        dim: usize,
        n_queries: usize,
        n_vectors: usize,
        device_id: usize,
        precision: Precision,
    ) -> Result<Vec<f32>, GpuError>;

    fn memory_usage(device_id: usize) -> Result<(usize, usize), GpuError>;
    fn device_count() -> usize;
}

pub fn l2_distance_gpu(
    queries: &[f32],
    corpus: &[f32],
    dim: usize,
    n_queries: usize,
    n_vectors: usize,
    device_id: usize,
    precision: Precision,
) -> Result<Vec<f32>, GpuError> {
    #[cfg(feature = "cuda")]
    {
        let device_count = cuda::device_count();
        if device_id >= device_count {
            return Err(GpuError::DeviceIndex(device_id));
        }

        let device_count = rocm::device_count();
        if device_id >= device_count {
            return Err(GpuError::DeviceIndex(device_id));
        }
        if !cuda::CudaBackend::supports_precision(precision) {
            return Err(GpuError::UnsupportedPrecision);
        }
        return cuda::CudaBackend::l2_distance(
            queries, corpus, dim, n_queries, n_vectors, device_id, precision
        );
    }
    
    #[cfg(feature = "rocm")]
    {
        if device_id >= rocm::device_count() {
            return Err(GpuError::DeviceIndex(device_id));
        }
        if !rocm::RocmBackend::supports_precision(precision) {
            return Err(GpuError::UnsupportedPrecision);
        }
        return rocm::RocmBackend::l2_distance(
            queries, corpus, dim, n_queries, n_vectors, device_id, precision
        );
    }
    
    Err(GpuError::NoBackend)
}

pub use device::set_active_device;
pub use precision::Precision;
pub use memory::GpuMemoryPool;