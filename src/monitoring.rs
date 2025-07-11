use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::sync::RwLock;
use std::io::{Read, Write};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Simplified metrics collector for monitoring ANN index performance
#[derive(Debug, Clone)]
pub struct MetricsCollector {
    /// Total number of queries processed
    pub query_count: Arc<AtomicU64>,
    /// Total query latency in microseconds
    pub total_query_latency_us: Arc<AtomicU64>,
    /// Index size (number of vectors)
    pub index_size: Arc<AtomicUsize>,
    /// Number of dimensions
    pub dimensions: Arc<AtomicUsize>,
    /// Distance metric name
    pub distance_metric: Arc<RwLock<String>>,
    /// Start time for uptime calculation
    pub start_time: Instant,
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            query_count: Arc::new(AtomicU64::new(0)),
            total_query_latency_us: Arc::new(AtomicU64::new(0)),
            index_size: Arc::new(AtomicUsize::new(0)),
            dimensions: Arc::new(AtomicUsize::new(0)),
            distance_metric: Arc::new(RwLock::new("unknown".to_string())),
            start_time: Instant::now(),
        }
    }

    /// Record a query with its latency
    pub fn record_query(&self, latency: Duration) {
        let latency_us = latency.as_micros() as u64;
        
        // Update counters
        self.query_count.fetch_add(1, Ordering::Relaxed);
        self.total_query_latency_us.fetch_add(latency_us, Ordering::Relaxed);
    }

    /// Set index metadata
    pub fn set_index_metadata(&self, size: usize, dims: usize, metric: &str) {
        self.index_size.store(size, Ordering::Relaxed);
        self.dimensions.store(dims, Ordering::Relaxed);
        if let Ok(mut distance_metric) = self.distance_metric.write() {
            *distance_metric = metric.to_string();
        }
    }

    /// Get current metrics snapshot (simplified version)
    pub fn get_metrics_snapshot(&self) -> MetricsSnapshot {
        let query_count = self.query_count.load(Ordering::Relaxed);
        let total_latency_us = self.total_query_latency_us.load(Ordering::Relaxed);
        let avg_latency_us = if query_count > 0 {
            total_latency_us as f64 / query_count as f64
        } else {
            0.0
        };

        let distance_metric = self.distance_metric.read()
            .map(|metric| metric.clone())
            .unwrap_or_else(|_| "unknown".to_string());

        MetricsSnapshot {
            query_count,
            avg_query_latency_us: avg_latency_us,
            index_size: self.index_size.load(Ordering::Relaxed),
            dimensions: self.dimensions.load(Ordering::Relaxed),
            distance_metric,
            uptime_seconds: self.start_time.elapsed().as_secs(),
        }
    }

    /// Update recall estimate (simplified - just a placeholder)
    pub fn update_recall_estimate(&self, _k: usize, _recall: f64) {
        // In the simplified version, we don't store recall estimates
        // This is just a placeholder to maintain API compatibility
    }

    /// Generate Prometheus format metrics
    pub fn to_prometheus_format(&self) -> String {
        let query_count = self.query_count.load(Ordering::Relaxed);
        let total_latency_us = self.total_query_latency_us.load(Ordering::Relaxed);
        let avg_latency_us = if query_count > 0 {
            total_latency_us as f64 / query_count as f64
        } else {
            0.0
        };

        let index_size = self.index_size.load(Ordering::Relaxed);
        let dimensions = self.dimensions.load(Ordering::Relaxed);
        let distance_metric = self.distance_metric.read()
            .map(|metric| metric.clone())
            .unwrap_or_else(|_| "unknown".to_string());
        let uptime_seconds = self.start_time.elapsed().as_secs();

        let mut output = String::new();

        // Basic metrics
        output.push_str("# HELP annie_queries_total Total number of queries processed\n");
        output.push_str("# TYPE annie_queries_total counter\n");
        output.push_str(&format!("annie_queries_total {}\n", query_count));

        output.push_str("# HELP annie_query_latency_microseconds_avg Average query latency in microseconds\n");
        output.push_str("# TYPE annie_query_latency_microseconds_avg gauge\n");
        output.push_str(&format!("annie_query_latency_microseconds_avg {:.2}\n", avg_latency_us));

        output.push_str("# HELP annie_index_size Number of vectors in the index\n");
        output.push_str("# TYPE annie_index_size gauge\n");
        output.push_str(&format!("annie_index_size {}\n", index_size));

        output.push_str("# HELP annie_dimensions Number of dimensions per vector\n");
        output.push_str("# TYPE annie_dimensions gauge\n");
        output.push_str(&format!("annie_dimensions {}\n", dimensions));

        output.push_str("# HELP annie_uptime_seconds Uptime in seconds\n");
        output.push_str("# TYPE annie_uptime_seconds gauge\n");
        output.push_str(&format!("annie_uptime_seconds {}\n", uptime_seconds));

        output.push_str("# HELP annie_distance_metric Distance metric used\n");
        output.push_str("# TYPE annie_distance_metric gauge\n");
        output.push_str(&format!("annie_distance_metric{{metric=\"{}\"}} 1\n", distance_metric));

        output
    }
}

/// Snapshot of metrics at a point in time (simplified version)
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub query_count: u64,
    pub avg_query_latency_us: f64,
    pub index_size: usize,
    pub dimensions: usize,
    pub distance_metric: String,
    pub uptime_seconds: u64,
}

/// Simple HTTP server for metrics endpoint
pub struct MetricsServer {
    metrics: Arc<MetricsCollector>,
    port: u16,
}

impl MetricsServer {
    pub fn new(metrics: Arc<MetricsCollector>, port: u16) -> Self {
        Self { metrics, port }
    }

    /// Start the metrics server in a background thread
    pub fn start(&self) -> std::io::Result<()> {
        use std::thread;
        use std::net::TcpListener;

        let listener = TcpListener::bind(format!("127.0.0.1:{}", self.port))?;
        let metrics = Arc::clone(&self.metrics);

        thread::spawn(move || {
            for stream in listener.incoming() {
                match stream {
                    Ok(stream) => {
                        let metrics_clone = Arc::clone(&metrics);
                        thread::spawn(move || {
                            if let Err(e) = handle_request(stream, metrics_clone) {
                                eprintln!("Error handling metrics request: {}", e);
                            }
                        });
                    }
                    Err(e) => {
                        eprintln!("Error accepting connection: {}", e);
                    }
                }
            }
        });

        Ok(())
    }
}

fn handle_request(mut stream: std::net::TcpStream, metrics: Arc<MetricsCollector>) -> std::io::Result<()> {
    let mut buffer = [0; 1024];
    stream.read(&mut buffer)?;

    let request = String::from_utf8_lossy(&buffer[..]);
    
    // Simple HTTP request parsing
    if request.starts_with("GET /metrics") {
        let metrics_output = metrics.to_prometheus_format();
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4; charset=utf-8\r\nContent-Length: {}\r\n\r\n{}",
            metrics_output.len(),
            metrics_output
        );
        stream.write_all(response.as_bytes())?;
    } else if request.starts_with("GET /health") {
        let response = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 2\r\n\r\nOK";
        stream.write_all(response.as_bytes())?;
    } else {
        let response = "HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\nContent-Length: 9\r\n\r\nNot Found";
        stream.write_all(response.as_bytes())?;
    }

    Ok(())
}

/// Python-facing metrics interface
#[pyclass]
pub struct PyMetricsCollector {
    inner: Arc<MetricsCollector>,
    server: Option<MetricsServer>,
}

#[pymethods]
impl PyMetricsCollector {
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(MetricsCollector::new()),
            server: None,
        }
    }

    /// Enable metrics endpoint on specified port
    fn enable_metrics(&mut self, port: u16) -> PyResult<()> {
        let server = MetricsServer::new(Arc::clone(&self.inner), port);
        server.start().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to start metrics server: {}", e)
        ))?;
        self.server = Some(server);
        Ok(())
    }

    /// Get current metrics as a Python dictionary
    fn get_metrics(&self, py: Python) -> PyResult<PyObject> {
        let metrics = &self.inner;
        let query_count = metrics.query_count.load(Ordering::Relaxed);
        let total_latency_us = metrics.total_query_latency_us.load(Ordering::Relaxed);
        let avg_latency_us = if query_count > 0 {
            total_latency_us as f64 / query_count as f64
        } else {
            0.0
        };

        let dict = PyDict::new(py);
        dict.set_item("query_count", query_count)?;
        dict.set_item("avg_query_latency_us", avg_latency_us)?;
        dict.set_item("index_size", metrics.index_size.load(Ordering::Relaxed))?;
        dict.set_item("dimensions", metrics.dimensions.load(Ordering::Relaxed))?;
        dict.set_item("uptime_seconds", metrics.start_time.elapsed().as_secs())?;
        
        let distance_metric = metrics.distance_metric.read()
            .map(|metric| metric.clone())
            .unwrap_or_else(|_| "unknown".to_string());
        dict.set_item("distance_metric", distance_metric)?;
        
        Ok(dict.into())
    }

    /// Get metrics in Prometheus format
    fn get_prometheus_metrics(&self) -> String {
        self.inner.to_prometheus_format()
    }
}

// Internal functions for use by the index implementations
impl PyMetricsCollector {
    pub fn get_collector(&self) -> Arc<MetricsCollector> {
        Arc::clone(&self.inner)
    }
}
