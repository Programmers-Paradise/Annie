use crate::gpu::{l2_distance_gpu, Precision, GpuError, set_active_device};
use super::ann_backend::AnnBackend;

pub struct GpuIndex {
    vectors: Vec<Vec<f32>>,
    dims: usize,
    device_id: usize,
    precision: Precision,
    memory_pool: crate::gpu::GpuMemoryPool,
}

impl GpuIndex {
    pub fn new(dims: usize, _distance: Distance) -> Self {
        Self {
            vectors: Vec::new(),
            dims,
            device_id: 0,
            precision: Precision::Fp32,
            memory_pool: crate::gpu::GpuMemoryPool::new(),
        }
    }
    
    pub fn set_device(&mut self, device_id: usize) -> Result<(), GpuError> {
        set_active_device(device_id)?;
        self.device_id = device_id;
        Ok(())
    }
    
    pub fn set_precision(&mut self, precision: Precision) {
        self.precision = precision;
    }
    
    pub fn memory_usage(&self) -> Option<(usize, usize)> {
        self.memory_pool.memory_usage(self.device_id)
    }
}

impl AnnBackend for GpuIndex {
    fn add(&mut self, vector: Vec<f32>) {
        self.vectors.push(vector);
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        if query.len() != self.dims {
            return Vec::new();
        }         
        // Convert vectors to flat array
        let corpus: Vec<f32> = self.vectors.iter().flatten().copied().collect();
        
        // Execute GPU search
        let distances = match l2_distance_gpu(
            query,
            &corpus,
            self.dims,
            1,
            self.vectors.len(),
            self.device_id,
            self.precision
        ) {
            Ok(d) => d,
            Err(_) => return Vec::new(),
        };
        
        // Process results
        let mut results: Vec<_> = distances.iter()
            .enumerate()
            .map(|(i, &d)| (i, d))
            .collect();
        
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
        results
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }
}