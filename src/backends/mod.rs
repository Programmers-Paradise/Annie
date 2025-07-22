pub mod ann_backend;
pub mod brute;
pub mod hnsw;
pub mod gpu;

use crate::metrics::Distance;
use ann_backend::AnnBackend;
use brute::BruteForceIndex;
use hnsw::HnswIndex;
use gpu::GpuIndex;
use crate::errors::BackendCreationError;

/// Enum to wrap the different backends under a single type.
pub enum BackendEnum {
    Brute(BruteForceIndex),
    Hnsw(HnswIndex),
    Gpu(GpuIndex),
}

impl BackendEnum {
    /// Create a new backend by type name.
    /// Returns a `Result` to handle cases where an unsupported backend is requested.
    pub fn new(backend_type: &str, dims: usize, distance: Distance) -> Result<Self, BackendCreationError> {
        match backend_type.to_lowercase().as_str() {
            "hnsw" => Ok(Self::Hnsw(HnswIndex::new(dims, distance))),
            "gpu" => Ok(Self::Gpu(GpuIndex::new(dims, distance))),
            "brute" | "bruteforce" => Ok(Self::Brute(BruteForceIndex::new(distance))),
            _      => Err(BackendCreationError::UnsupportedBackend(backend_type.to_string())),
        }
    }

    /// Sets the active GPU device if the backend is a GpuIndex.
    pub fn set_gpu_device(&mut self, device_id: usize) -> Result<(), crate::gpu::GpuError> {
        if let BackendEnum::Gpu(gpu) = self {
            gpu.set_device(device_id)
        } else {
            // Return an error if this is called on a non-GPU backend.
            Err(crate::gpu::GpuError::NoBackend)
        }
    }
    
    /// Gets the memory usage if the backend is a GpuIndex.
    pub fn gpu_memory_usage(&self) -> Option<(usize, usize)> {
        if let BackendEnum::Gpu(gpu) = self {
            gpu.memory_usage()
        } else {
            None
        }
    }
}

impl AnnBackend for BackendEnum {
    fn add(&mut self, vector: Vec<f32>) {
        match self {
            BackendEnum::Brute(b) => b.add(vector),
            BackendEnum::Hnsw(h)  => h.add(vector),
            BackendEnum::Gpu(g)   => g.add(vector),
        }
    }
    fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        match self {
            BackendEnum::Brute(b) => b.search(query, k),
            BackendEnum::Hnsw(h)  => h.search(query, k),
            BackendEnum::Gpu(g)   => g.search(query, k),
        }
    }
    fn len(&self) -> usize {
        match self {
            BackendEnum::Brute(b) => b.len(),
            BackendEnum::Hnsw(h)  => h.len(),
            BackendEnum::Gpu(g)   => g.len(),
        }
    }
}