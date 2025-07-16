pub mod ann_backend;
pub mod brute;
pub mod hnsw;
pub mod gpu;

use crate::metrics::Distance;
use ann_backend::AnnBackend;
use brute::BruteForceIndex;
use hnsw::HnswIndex;
use gpu::GpuIndex;

/// Enum to wrap the different backends under a single type.
pub enum BackendEnum {
    Brute(BruteForceIndex),
    Hnsw(HnswIndex),
    Gpu(GpuIndex),
}

impl BackendEnum {
    /// Create a new backend by type name.
    pub fn new(backend_type: &str, dims: usize, distance: Distance) -> Self {
        match backend_type {
            "hnsw" => Self::Hnsw(HnswIndex::new(dims, distance)),
            "gpu" => Self::Gpu(GpuIndex::new(dims, distance)),
            _      => Self::Brute(BruteForceIndex::new(distance)),
        }
    }
}

impl BackendEnum {
    pub fn set_gpu_device(&mut self, device_id: usize) -> Result<(), crate::gpu::GpuError> {
        if let BackendEnum::Gpu(gpu) = self {
            gpu.set_device(device_id)
        } else {
            Err(crate::gpu::GpuError::DeviceIndex(0))
        }
    }
    
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
        }
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        match self {
            BackendEnum::Brute(b) => b.search(query, k),
            BackendEnum::Hnsw(h)  => h.search(query, k),
        }
    }

    fn len(&self) -> usize {
        match self {
            BackendEnum::Brute(b) => b.len(),
            BackendEnum::Hnsw(h)  => h.len(),
        }
    }
}
