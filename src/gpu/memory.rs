use crate::gpu::{GpuError, Precision};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// RAII wrapper for GPU memory buffer with automatic cleanup
pub struct GpuBuffer {
    buffer: Vec<u8>,
    device_id: usize,
    size: usize,
    precision: Precision,
    pool: Arc<Mutex<GpuMemoryPool>>,
    returned: bool,
}

impl GpuBuffer {
    fn new(
        buffer: Vec<u8>,
        device_id: usize,
        size: usize,
        precision: Precision,
        pool: Arc<Mutex<GpuMemoryPool>>,
    ) -> Self {
        Self {
            buffer,
            device_id,
            size,
            precision,
            pool,
            returned: false,
        }
    }

    /// Get a reference to the underlying buffer
    pub fn as_slice(&self) -> &[u8] {
        &self.buffer
    }

    /// Get a mutable reference to the underlying buffer
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.buffer
    }

    /// Get the buffer size
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Manually return the buffer to the pool (normally done automatically on drop)
    pub fn return_to_pool(&mut self) {
        if !self.returned {
            if let Ok(mut pool) = self.pool.lock() {
                pool.return_buffer(self.device_id, std::mem::take(&mut self.buffer), self.size, self.precision);
                self.returned = true;
            }
        }
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        self.return_to_pool();
    }
}

/// RAII wrapper for multiple GPU buffers with batch cleanup
pub struct GpuBufferBatch {
    buffers: Vec<GpuBuffer>,
    device_id: usize,
    pool: Arc<Mutex<GpuMemoryPool>>,
}

impl GpuBufferBatch {
    pub fn new(device_id: usize, pool: Arc<Mutex<GpuMemoryPool>>) -> Self {
        Self {
            buffers: Vec::new(),
            device_id,
            pool,
        }
    }

    /// Add a buffer to the batch
    pub fn add_buffer(&mut self, buffer: Vec<u8>, size: usize, precision: Precision) {
        let gpu_buffer = GpuBuffer::new(buffer, self.device_id, size, precision, self.pool.clone());
        self.buffers.push(gpu_buffer);
    }

    /// Get a reference to a buffer by index
    pub fn get_buffer(&self, index: usize) -> Option<&GpuBuffer> {
        self.buffers.get(index)
    }

    /// Get a mutable reference to a buffer by index
    pub fn get_buffer_mut(&mut self, index: usize) -> Option<&mut GpuBuffer> {
        self.buffers.get_mut(index)
    }

    /// Get the number of buffers in the batch
    pub fn len(&self) -> usize {
        self.buffers.len()
    }

    /// Check if the batch is empty
    pub fn is_empty(&self) -> bool {
        self.buffers.is_empty()
    }

    /// Manually return all buffers to the pool
    pub fn return_all_to_pool(&mut self) {
        for buffer in &mut self.buffers {
            buffer.return_to_pool();
        }
    }
}

impl Drop for GpuBufferBatch {
    fn drop(&mut self) {
        self.return_all_to_pool();
    }
}

struct DeviceMemoryPool {
    buffers: HashMap<(usize, Precision), Vec<Vec<u8>>>,
    allocated: usize,
    peak_usage: usize,
    max_pool_size: usize,
    fragmentation_threshold: f32,
}

impl DeviceMemoryPool {
    fn new() -> Self {
        Self {
            buffers: HashMap::new(),
            allocated: 0,
            peak_usage: 0,
            max_pool_size: 1024 * 1024 * 1024, // 1GB default max pool size
            fragmentation_threshold: 0.5, // Cleanup when 50% fragmented
        }
    }

    fn get_buffer(&mut self, size: usize, precision: Precision) -> Vec<u8> {
        let key = (size, precision);
        if let Some(buffers) = self.buffers.get_mut(&key) {
            if let Some(buf) = buffers.pop() {
                self.record_allocation(buf.len());
                return buf;
            }
        }
        
        let elem_size = match precision {
            Precision::Fp32 => 4,
            Precision::Fp16 => 2,
            Precision::Int8 => 1,
        };
        let bytes = size * elem_size;
        
        // Check memory pressure before allocation
        if self.allocated + bytes > self.max_pool_size {
            self.cleanup_fragmented_buffers();
        }
        
        self.record_allocation(bytes);
        vec![0u8; bytes]
    }

    fn return_buffer(&mut self, buffer: Vec<u8>, size: usize, precision: Precision) {
        let key = (size, precision);
        let buffer_size = buffer.len();
        
        // Check if we should keep this buffer or drop it due to memory pressure
        if self.allocated < self.max_pool_size {
            self.buffers.entry(key).or_insert_with(Vec::new).push(buffer);
        }
        
        self.record_deallocation(buffer_size);
    }

    fn record_allocation(&mut self, bytes: usize) {
        self.allocated += bytes;
        self.peak_usage = self.peak_usage.max(self.allocated);
    }

    fn record_deallocation(&mut self, bytes: usize) {
        if bytes > self.allocated {
            self.allocated = 0;
        } else {
            self.allocated -= bytes;
        }
    }

    /// Clean up fragmented buffers to reduce memory pressure
    fn cleanup_fragmented_buffers(&mut self) {
        let total_buffers: usize = self.buffers.values().map(|v| v.len()).sum();
        if total_buffers == 0 {
            return;
        }

        let buffer_types = self.buffers.len();
        let avg_buffers_per_type = total_buffers / buffer_types.max(1);
        let fragmentation_ratio = buffer_types as f32 / total_buffers as f32;

        if fragmentation_ratio > self.fragmentation_threshold {
            // Remove buffer types with only a few buffers
            self.buffers.retain(|_, buffers| buffers.len() > avg_buffers_per_type / 2);
            
            // Limit number of buffers per type
            for buffers in self.buffers.values_mut() {
                buffers.truncate(avg_buffers_per_type * 2);
            }
        }
    }

    /// Emergency cleanup - remove all cached buffers
    fn emergency_cleanup(&mut self) {
        self.buffers.clear();
        self.allocated = 0;
    }

    /// Get memory pressure ratio (0.0 = no pressure, 1.0 = at limit)
    fn memory_pressure(&self) -> f32 {
        self.allocated as f32 / self.max_pool_size as f32
    }

    /// Set maximum pool size
    fn set_max_pool_size(&mut self, max_size: usize) {
        self.max_pool_size = max_size;
        if self.allocated > max_size {
            self.cleanup_fragmented_buffers();
        }
    }
}

#[derive(Clone)]
pub struct GpuMemoryPool(Arc<Mutex<HashMap<usize, Arc<Mutex<DeviceMemoryPool>>>>>);

impl GpuMemoryPool {
    pub fn new() -> Self {
        Self(Arc::new(Mutex::new(HashMap::new())))
    }

    pub fn get_buffer(&self, device_id: usize, size: usize, precision: Precision) -> Vec<u8> {
        let pool = {
            let mut pools = self.0.lock().unwrap();
            pools.entry(device_id).or_insert_with(|| Arc::new(Mutex::new(DeviceMemoryPool::new()))).clone()
        };
        let mut pool = pool.lock().unwrap();
        pool.get_buffer(size, precision)
    }

    /// Get a RAII-wrapped buffer that automatically returns to pool on drop
    pub fn get_managed_buffer(&self, device_id: usize, size: usize, precision: Precision) -> GpuBuffer {
        let buffer = self.get_buffer(device_id, size, precision);
        GpuBuffer::new(buffer, device_id, size, precision, Arc::new(Mutex::new(self.clone())))
    }

    /// Create a batch of managed buffers
    pub fn create_buffer_batch(&self, device_id: usize) -> GpuBufferBatch {
        GpuBufferBatch::new(device_id, Arc::new(Mutex::new(self.clone())))
    }

    pub fn return_buffer(&self, device_id: usize, buffer: Vec<u8>, size: usize, precision: Precision) {
        if let Some(pool) = {
            let pools = self.0.lock().unwrap();
            pools.get(&device_id).cloned()
        } {
            let mut pool = pool.lock().unwrap();
            pool.return_buffer(buffer, size, precision);
        }
    }

    pub fn memory_usage(&self, device_id: usize) -> Option<(usize, usize)> {
        let pool = {
            let pools = self.0.lock().unwrap();
            pools.get(&device_id).cloned()
        }?;
        let pool = pool.lock().unwrap();
        Some((pool.allocated, pool.peak_usage))
    }

    /// Get memory pressure for a device (0.0 = no pressure, 1.0 = at limit)
    pub fn memory_pressure(&self, device_id: usize) -> Option<f32> {
        let pool = {
            let pools = self.0.lock().unwrap();
            pools.get(&device_id).cloned()
        }?;
        let pool = pool.lock().unwrap();
        Some(pool.memory_pressure())
    }

    /// Set maximum pool size for a device
    pub fn set_max_pool_size(&self, device_id: usize, max_size: usize) {
        let pool = {
            let mut pools = self.0.lock().unwrap();
            pools.entry(device_id).or_insert_with(|| Arc::new(Mutex::new(DeviceMemoryPool::new()))).clone()
        };
        let mut pool = pool.lock().unwrap();
        pool.set_max_pool_size(max_size);
    }

    /// Emergency cleanup for a device - removes all cached buffers
    pub fn emergency_cleanup(&self, device_id: usize) {
        if let Some(pool) = {
            let pools = self.0.lock().unwrap();
            pools.get(&device_id).cloned()
        } {
            let mut pool = pool.lock().unwrap();
            pool.emergency_cleanup();
        }
    }

    /// Emergency cleanup for all devices
    pub fn emergency_cleanup_all(&self) {
        let pools = self.0.lock().unwrap();
        for pool in pools.values() {
            let mut pool = pool.lock().unwrap();
            pool.emergency_cleanup();
        }
    }

    /// Get total memory usage across all devices
    pub fn total_memory_usage(&self) -> (usize, usize) {
        let pools = self.0.lock().unwrap();
        let mut total_allocated = 0;
        let mut total_peak = 0;
        
        for pool in pools.values() {
            let pool = pool.lock().unwrap();
            total_allocated += pool.allocated;
            total_peak += pool.peak_usage;
        }
        
        (total_allocated, total_peak)
    }
}