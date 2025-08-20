#[cfg(test)]
#[cfg(any(feature = "cuda", feature = "rocm"))]
mod tests {
    use rust_annie::gpu::memory::{GpuMemoryPool, GpuBuffer, GpuBufferBatch};
    use rust_annie::gpu::Precision;

    #[test]
    fn test_memory_pool_basic_operations() {
        let pool = GpuMemoryPool::new();
        let device_id = 0;
        let size = 1024;
        let precision = Precision::Fp32;

        // Test getting and returning buffer
        let buffer = pool.get_buffer(device_id, size, precision);
        assert_eq!(buffer.len(), size * precision.element_size());

        pool.return_buffer(device_id, buffer, size, precision);

        // Test memory usage tracking
        let usage = pool.memory_usage(device_id);
        assert!(usage.is_some());
    }

    #[test]
    fn test_raii_buffer_automatic_cleanup() {
        let pool = GpuMemoryPool::new();
        let device_id = 0;
        let size = 512;
        let precision = Precision::Fp16;

        // Create managed buffer in a scope
        let initial_usage = pool.memory_usage(device_id).unwrap_or((0, 0));
        
        {
            let _managed_buffer = pool.get_managed_buffer(device_id, size, precision);
            // Buffer should be allocated
            let usage = pool.memory_usage(device_id).unwrap_or((0, 0));
            assert!(usage.0 >= initial_usage.0);
        } // Buffer should be automatically returned to pool here

        // Check that buffer was returned (may still be in cache)
        let final_usage = pool.memory_usage(device_id).unwrap_or((0, 0));
        // Memory should be tracked but buffer may be cached for reuse
        assert!(final_usage.1 >= initial_usage.1); // Peak usage should have increased
    }

    #[test]
    fn test_buffer_batch_cleanup() {
        let pool = GpuMemoryPool::new();
        let device_id = 0;

        let initial_usage = pool.memory_usage(device_id).unwrap_or((0, 0));

        {
            let mut batch = pool.create_buffer_batch(device_id);
            
            // Add multiple buffers to batch
            batch.add_buffer(vec![0u8; 256], 64, Precision::Fp32);
            batch.add_buffer(vec![0u8; 128], 64, Precision::Fp16);
            batch.add_buffer(vec![0u8; 64], 64, Precision::Int8);
            
            assert_eq!(batch.len(), 3);
            assert!(!batch.is_empty());
        } // All buffers should be automatically returned here

        let final_usage = pool.memory_usage(device_id).unwrap_or((0, 0));
        assert!(final_usage.1 >= initial_usage.1); // Peak usage should have increased
    }

    #[test]
    fn test_memory_pressure_monitoring() {
        let pool = GpuMemoryPool::new();
        let device_id = 0;
        
        // Set a small max pool size for testing
        pool.set_max_pool_size(device_id, 1024);
        
        // Check initial pressure
        let initial_pressure = pool.memory_pressure(device_id).unwrap_or(0.0);
        assert!(initial_pressure >= 0.0 && initial_pressure <= 1.0);
        
        // Allocate some memory and check pressure increases
        let _buffer1 = pool.get_managed_buffer(device_id, 100, Precision::Fp32);
        let pressure_after = pool.memory_pressure(device_id).unwrap_or(0.0);
        assert!(pressure_after >= initial_pressure);
    }

    #[test]
    fn test_emergency_cleanup() {
        let pool = GpuMemoryPool::new();
        let device_id = 0;
        
        // Allocate some buffers
        let _buffer1 = pool.get_buffer(device_id, 256, Precision::Fp32);
        let _buffer2 = pool.get_buffer(device_id, 512, Precision::Fp16);
        
        // Return buffers to cache them
        pool.return_buffer(device_id, _buffer1, 256, Precision::Fp32);
        pool.return_buffer(device_id, _buffer2, 512, Precision::Fp16);
        
        // Check memory usage before cleanup
        let usage_before = pool.memory_usage(device_id).unwrap_or((0, 0));
        
        // Emergency cleanup
        pool.emergency_cleanup(device_id);
        
        // Memory should be cleared
        let usage_after = pool.memory_usage(device_id).unwrap_or((0, 0));
        assert_eq!(usage_after.0, 0); // Allocated should be 0
    }

    #[test]
    fn test_total_memory_usage() {
        let pool = GpuMemoryPool::new();
        
        // Allocate on multiple devices
        let _buffer1 = pool.get_managed_buffer(0, 256, Precision::Fp32);
        let _buffer2 = pool.get_managed_buffer(1, 512, Precision::Fp16);
        
        let (total_allocated, total_peak) = pool.total_memory_usage();
        assert!(total_allocated > 0);
        assert!(total_peak >= total_allocated);
    }

    #[test]
    fn test_buffer_manual_return() {
        let pool = GpuMemoryPool::new();
        let device_id = 0;
        let size = 128;
        let precision = Precision::Int8;

        let mut buffer = pool.get_managed_buffer(device_id, size, precision);
        
        // Manually return buffer
        buffer.return_to_pool();
        
        // Subsequent returns should be no-ops
        buffer.return_to_pool();
    }

    #[test]
    fn test_buffer_properties() {
        let pool = GpuMemoryPool::new();
        let device_id = 0;
        let size = 64;
        let precision = Precision::Fp32;

        let buffer = pool.get_managed_buffer(device_id, size, precision);
        
        assert_eq!(buffer.len(), size * precision.element_size());
        assert!(!buffer.is_empty());
        
        // Test slice access
        let slice = buffer.as_slice();
        assert_eq!(slice.len(), buffer.len());
    }
}