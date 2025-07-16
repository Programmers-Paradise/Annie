use crate::gpu::GpuError;

static ACTIVE_DEVICE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub fn set_active_device(device_id: usize) -> Result<(), GpuError> {
    #[cfg(feature = "cuda")]
    {
        if device_id >= cuda::device_count() {
            return Err(GpuError::DeviceIndex(device_id));
        }
        cust::device::set_device(device_id as u32).map_err(GpuError::Cuda)?;
    }
    
    #[cfg(feature = "rocm")]
    {
        if device_id >= rocm::device_count() {
            return Err(GpuError::DeviceIndex(device_id));
        }
        hip_runtime::device::set_device(device_id as u32).map_err(|e| GpuError::Rocm(e.into()))?;
    }
    
    #[cfg(not(any(feature = "cuda", feature = "rocm")))]
    {
        return Err(GpuError::NoBackend);
    }
    
    ACTIVE_DEVICE.store(device_id, std::sync::atomic::Ordering::SeqCst);
    Ok(())
}

pub fn get_active_device() -> usize {
    ACTIVE_DEVICE.load(std::sync::atomic::Ordering::SeqCst)
}