#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "rocm")]
mod rocm;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum GpuError {
    #[error("No compatible GPU backend found")]
    NoBackend,
    #[error("CUDA error: {0}")]
    Cuda(#[from] cust::error::CudaError),
    #[error("ROCm error: {0}")]
    Rocm(#[from] hip_runtime::Status),
}

pub trait GpuBackend {
    fn l2_distance(
        queries: &[f32],
        corpus: &[f32],
        dim: usize,
        n_queries: usize,
        n_vectors: usize,
    ) -> Result<Vec<f32>, GpuError>;
}

pub fn l2_distance_gpu(
    queries: &[f32],
    corpus: &[f32],
    dim: usize,
    n_queries: usize,
    n_vectors: usize,
) -> Result<Vec<f32>, GpuError> {
    #[cfg(feature = "cuda")]
    {
        return cuda::CudaBackend::l2_distance(queries, corpus, dim, n_queries, n_vectors);
    }
    
    #[cfg(feature = "rocm")]
    {
        return rocm::RocmBackend::l2_distance(queries, corpus, dim, n_queries, n_vectors);
    }
    
    Err(GpuError::NoBackend)
}