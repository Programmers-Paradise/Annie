use crate::gpu::memory::{GpuMemoryPool, MemoryStats};
use std::time::{Duration, Instant};
use std::thread;
use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};

/// Memory usage monitoring and reporting utility
pub struct MemoryMonitor {
    pool: Arc<Mutex<GpuMemoryPool>>,
    running: Arc<AtomicBool>,
    report_interval: Duration,
    cleanup_interval: Duration,
    memory_pressure_threshold: f32,
}

/// Memory report containing usage statistics
#[derive(Debug, Clone)]
pub struct MemoryReport {
    pub timestamp: Instant,
    pub total_allocated: usize,
    pub total_peak: usize,
    pub device_stats: Vec<(usize, MemoryStats)>,
    pub average_cache_efficiency: f32,
    pub memory_pressure_alerts: Vec<usize>, // Device IDs with high pressure
}

impl MemoryMonitor {
    /// Create a new memory monitor
    pub fn new(
        pool: Arc<Mutex<GpuMemoryPool>>,
        report_interval: Duration,
        cleanup_interval: Duration,
    ) -> Self {
        Self {
            pool,
            running: Arc::new(AtomicBool::new(false)),
            report_interval,
            cleanup_interval,
            memory_pressure_threshold: 0.8, // 80% threshold for alerts
        }
    }

    /// Set memory pressure threshold (0.0 - 1.0)
    pub fn set_pressure_threshold(&mut self, threshold: f32) {
        self.memory_pressure_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Start monitoring in background thread
    pub fn start_monitoring(&self) -> thread::JoinHandle<()> {
        self.running.store(true, Ordering::SeqCst);
        let pool = self.pool.clone();
        let running = self.running.clone();
        let report_interval = self.report_interval;
        let cleanup_interval = self.cleanup_interval;
        let pressure_threshold = self.memory_pressure_threshold;

        thread::spawn(move || {
            let mut last_cleanup = Instant::now();
            
            while running.load(Ordering::SeqCst) {
                thread::sleep(report_interval);

                if let Ok(pool_guard) = pool.lock() {
                    // Generate memory report
                    let report = Self::generate_report(&*pool_guard, pressure_threshold);
                    Self::log_report(&report);

                    // Handle memory pressure alerts
                    if !report.memory_pressure_alerts.is_empty() {
                        Self::handle_pressure_alerts(&*pool_guard, &report.memory_pressure_alerts);
                    }

                    // Perform maintenance cleanup if needed
                    if last_cleanup.elapsed() >= cleanup_interval {
                        pool_guard.maintenance_cleanup(cleanup_interval);
                        last_cleanup = Instant::now();
                    }
                }
            }
        })
    }

    /// Stop monitoring
    pub fn stop_monitoring(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    /// Generate a one-time memory report
    pub fn generate_report(pool: &GpuMemoryPool, pressure_threshold: f32) -> MemoryReport {
        let (total_allocated, total_peak) = pool.total_memory_usage();
        let summary = pool.summary_stats();
        
        let mut device_stats = Vec::new();
        let mut total_efficiency = 0.0;
        let mut efficiency_count = 0;
        let mut pressure_alerts = Vec::new();

        for (device_id, stats) in summary {
            device_stats.push((device_id, stats));
            
            if let Some(efficiency) = pool.cache_efficiency(device_id) {
                total_efficiency += efficiency;
                efficiency_count += 1;
            }

            if let Some(pressure) = pool.memory_pressure(device_id) {
                if pressure > pressure_threshold {
                    pressure_alerts.push(device_id);
                }
            }
        }

        let average_cache_efficiency = if efficiency_count > 0 {
            total_efficiency / efficiency_count as f32
        } else {
            0.0
        };

        MemoryReport {
            timestamp: Instant::now(),
            total_allocated,
            total_peak,
            device_stats,
            average_cache_efficiency,
            memory_pressure_alerts: pressure_alerts,
        }
    }

    /// Log memory report
    fn log_report(report: &MemoryReport) {
        println!("=== GPU Memory Report ===");
        println!("Total Allocated: {} bytes ({:.2} MB)", 
                 report.total_allocated, 
                 report.total_allocated as f64 / (1024.0 * 1024.0));
        println!("Total Peak: {} bytes ({:.2} MB)", 
                 report.total_peak, 
                 report.total_peak as f64 / (1024.0 * 1024.0));
        println!("Average Cache Efficiency: {:.1}%", 
                 report.average_cache_efficiency * 100.0);

        if !report.memory_pressure_alerts.is_empty() {
            println!("⚠️  Memory Pressure Alerts: {:?}", report.memory_pressure_alerts);
        }

        for (device_id, stats) in &report.device_stats {
            println!("Device {}: {} allocs, {} deallocs, {:.1}% cache efficiency",
                     device_id,
                     stats.allocation_count,
                     stats.deallocation_count,
                     (stats.cache_hits as f64 / (stats.cache_hits + stats.cache_misses).max(1) as f64) * 100.0);
        }
        println!("========================");
    }

    /// Handle memory pressure alerts
    fn handle_pressure_alerts(pool: &GpuMemoryPool, alert_devices: &[usize]) {
        for &device_id in alert_devices {
            if let Some(pressure) = pool.memory_pressure(device_id) {
                println!("🚨 High memory pressure on device {}: {:.1}%", 
                         device_id, pressure * 100.0);
                
                if pressure > 0.9 {
                    println!("   Performing emergency cleanup...");
                    pool.emergency_cleanup(device_id);
                }
            }
        }
    }

    /// Get current memory status
    pub fn get_current_report(&self) -> Option<MemoryReport> {
        self.pool.lock().ok().map(|pool| {
            Self::generate_report(&*pool, self.memory_pressure_threshold)
        })
    }
}

/// Convenience function to create and start a memory monitor
pub fn start_memory_monitoring(
    pool: Arc<Mutex<GpuMemoryPool>>,
    report_interval_secs: u64,
    cleanup_interval_secs: u64,
) -> (MemoryMonitor, thread::JoinHandle<()>) {
    let monitor = MemoryMonitor::new(
        pool,
        Duration::from_secs(report_interval_secs),
        Duration::from_secs(cleanup_interval_secs),
    );
    let handle = monitor.start_monitoring();
    (monitor, handle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::memory::GpuMemoryPool;
    use crate::gpu::Precision;

    #[test]
    fn test_memory_monitor_report_generation() {
        let pool = GpuMemoryPool::new();
        
        // Create some memory usage
        let _buffer1 = pool.get_managed_buffer(0, 256, Precision::Fp32);
        let _buffer2 = pool.get_managed_buffer(1, 512, Precision::Fp16);
        
        // Generate report
        let report = MemoryMonitor::generate_report(&pool, 0.8);
        
        assert!(report.total_allocated > 0);
        assert!(report.device_stats.len() >= 2);
        assert!(report.average_cache_efficiency >= 0.0);
        assert!(report.average_cache_efficiency <= 1.0);
    }

    #[test]
    fn test_memory_monitor_creation() {
        let pool = Arc::new(Mutex::new(GpuMemoryPool::new()));
        let monitor = MemoryMonitor::new(
            pool,
            Duration::from_secs(1),
            Duration::from_secs(60),
        );
        
        // Test that we can get a report
        let report = monitor.get_current_report();
        assert!(report.is_some());
    }
}