use crate::gpu::{GpuError, Precision};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

struct DeviceMemoryPool {
    buffers: HashMap<(usize, Precision), Vec<Vec<u8>>>,
    allocated: usize,
    peak_usage: usize,
}

impl DeviceMemoryPool {
    fn new() -> Self {
        Self {
            buffers: HashMap::new(),
            allocated: 0,
            peak_usage: 0,
        }
    }

    fn get_buffer(&mut self, size: usize, precision: Precision) -> Vec<u8> {
        let key = (size, precision);
        if let Some(buffers) = self.buffers.get_mut(&key) {
            if let Some(buf) = buffers.pop() {
                return buf;
            }
        }
        
        let elem_size = match precision {
            Precision::Fp32 => 4,
            Precision::Fp16 => 2,
            Precision::Int8 => 1,
        };
        let bytes = size * elem_size;
        vec![0u8; bytes]
    }

    fn return_buffer(&mut self, buffer: Vec<u8>, size: usize, precision: Precision) {
        let key = (size, precision);
        self.buffers.entry(key).or_insert_with(Vec::new).push(buffer);
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
}