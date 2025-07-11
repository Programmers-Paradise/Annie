# Annie Streaming K-NN Monitoring System

This document describes the streaming K-NN recall & latency monitoring system implemented in Annie.

## Overview

The monitoring system provides:
- **Real-time query latency tracking** - Monitor average query response times
- **Index statistics** - Track index size, dimensions, and configuration
- **Prometheus metrics endpoint** - Industry-standard metrics format
- **Low overhead design** - Minimal performance impact on queries
- **Python integration** - Easy to use in Python server contexts

## Quick Start

### 1. Enable Metrics on an Index

```python
import numpy as np
from rust_annie import AnnIndex, Distance

# Create an index
index = AnnIndex(128, Distance.EUCLIDEAN)

# Enable metrics with HTTP server on port 8000
index.enable_metrics(8000)

# Add some data
data = np.random.random((1000, 128)).astype(np.float32)
ids = np.arange(1000, dtype=np.int64)
index.add(data, ids)
```

### 2. Run Queries (Metrics Collected Automatically)

```python
# Metrics are automatically collected for each query
query = np.random.random(128).astype(np.float32)
labels, distances = index.search(query, k=10)

# Check current metrics
metrics = index.get_metrics()
print(f"Query count: {metrics['query_count']}")
print(f"Average latency: {metrics['avg_query_latency_us']} μs")
```

### 3. Access Metrics via HTTP

```bash
# Get Prometheus format metrics
curl http://localhost:8000/metrics

# Health check
curl http://localhost:8000/health
```

## API Reference

### AnnIndex Methods

#### `enable_metrics(port: int = None)`
Enable metrics collection for the index.

**Parameters:**
- `port` (int, optional): Port number for HTTP metrics server. If provided, starts an HTTP server.

**Example:**
```python
index.enable_metrics(8000)  # Start server on port 8000
index.enable_metrics()      # Enable metrics without HTTP server
```

#### `get_metrics() -> dict`
Get current metrics as a Python dictionary.

**Returns:**
- Dictionary containing current metrics

**Example:**
```python
metrics = index.get_metrics()
# Returns:
# {
#     'query_count': 150,
#     'avg_query_latency_us': 45.2,
#     'index_size': 1000,
#     'dimensions': 128,
#     'distance_metric': 'euclidean',
#     'uptime_seconds': 300,
#     'recall_estimates': {}
# }
```

#### `update_recall_estimate(k: int, recall: float)`
Update recall estimate for a specific k value.

**Parameters:**
- `k` (int): The k value for nearest neighbor search
- `recall` (float): Estimated recall value (0.0 to 1.0)

**Example:**
```python
index.update_recall_estimate(10, 0.95)  # 95% recall for k=10
```

### Standalone Metrics Collector

For advanced use cases, you can use the standalone metrics collector:

```python
from rust_annie import PyMetricsCollector

# Create standalone collector
collector = PyMetricsCollector()

# Enable HTTP server
collector.enable_metrics(8001)

# Get metrics
metrics = collector.get_metrics()

# Get Prometheus format
prometheus_text = collector.get_prometheus_metrics()
```

## Metrics Reference

### Available Metrics

#### Core Metrics
- **`annie_queries_total`** (counter): Total number of queries processed
- **`annie_query_latency_microseconds_avg`** (gauge): Average query latency in microseconds
- **`annie_index_size`** (gauge): Number of vectors in the index
- **`annie_dimensions`** (gauge): Number of dimensions per vector
- **`annie_uptime_seconds`** (gauge): Uptime in seconds
- **`annie_distance_metric`** (gauge): Distance metric used (labeled)

#### Example Prometheus Output

```
# HELP annie_queries_total Total number of queries processed
# TYPE annie_queries_total counter
annie_queries_total 1523

# HELP annie_query_latency_microseconds_avg Average query latency in microseconds
# TYPE annie_query_latency_microseconds_avg gauge
annie_query_latency_microseconds_avg 42.75

# HELP annie_index_size Number of vectors in the index
# TYPE annie_index_size gauge
annie_index_size 10000

# HELP annie_dimensions Number of dimensions per vector
# TYPE annie_dimensions gauge
annie_dimensions 128

# HELP annie_uptime_seconds Uptime in seconds
# TYPE annie_uptime_seconds gauge
annie_uptime_seconds 3600

# HELP annie_distance_metric Distance metric used
# TYPE annie_distance_metric gauge
annie_distance_metric{metric="euclidean"} 1
```

### HTTP Endpoints

#### `GET /metrics`
Returns metrics in Prometheus format.

**Response:**
- **Content-Type:** `text/plain; version=0.0.4; charset=utf-8`
- **Body:** Prometheus format metrics

#### `GET /health`
Health check endpoint.

**Response:**
- **Content-Type:** `text/plain`
- **Body:** `OK`

## Integration Examples

### With Flask

```python
from flask import Flask, jsonify
from rust_annie import AnnIndex, Distance
import numpy as np

app = Flask(__name__)

# Create index with metrics
index = AnnIndex(128, Distance.EUCLIDEAN)
index.enable_metrics(8001)  # Separate port for metrics

# Add data
data = np.random.random((10000, 128)).astype(np.float32)
ids = np.arange(10000, dtype=np.int64)
index.add(data, ids)

@app.route('/search')
def search():
    # Generate random query for demo
    query = np.random.random(128).astype(np.float32)
    labels, distances = index.search(query, k=10)
    
    return jsonify({
        'labels': labels.tolist(),
        'distances': distances.tolist()
    })

@app.route('/stats')
def stats():
    return jsonify(index.get_metrics())

if __name__ == '__main__':
    print("API server: http://localhost:5000")
    print("Metrics: http://localhost:8001/metrics")
    app.run(host='0.0.0.0', port=5000)
```

### With Prometheus Monitoring

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'annie'
    static_configs:
      - targets: ['localhost:8000']
    scrape_interval: 15s
    metrics_path: /metrics
```

### Performance Considerations

1. **Low Overhead**: Metrics collection adds minimal overhead (< 1μs per query)
2. **Atomic Operations**: Thread-safe using atomic operations
3. **No GIL Impact**: Metrics recording doesn't hold the Python GIL
4. **Background Server**: HTTP server runs in background threads

### Best Practices

1. **Choose Appropriate Ports**: Use non-conflicting ports for metrics servers
2. **Monitor Resource Usage**: Keep an eye on memory usage for long-running applications
3. **Regular Metrics Review**: Use metrics to optimize query performance and index configuration
4. **Integration with Alerting**: Set up alerts for high latency or error rates

## Troubleshooting

### Common Issues

**Port Already in Use**
```python
# If port 8000 is busy, try a different port
index.enable_metrics(8001)
```

**No Metrics Available**
```python
# Make sure metrics are enabled first
index.enable_metrics()
metrics = index.get_metrics()
```

**Server Not Responding**
```bash
# Check if the process is running and port is open
netstat -tlnp | grep 8000
```

### Debug Information

To get debug information about the metrics system:

```python
metrics = index.get_metrics()
print("Metrics available:", metrics is not None)
print("Query count:", metrics.get('query_count', 0))
print("Uptime:", metrics.get('uptime_seconds', 0))
```

## Conclusion

The Annie monitoring system provides production-ready metrics for K-NN workloads with minimal overhead. It integrates seamlessly with existing monitoring infrastructure through Prometheus-compatible endpoints and provides detailed insights into query performance and index characteristics.
