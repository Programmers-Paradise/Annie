"""
Test script for the streaming K-NN recall & latency monitoring feature.

This script demonstrates:
1. Enabling metrics collection on an index
2. Starting a metrics HTTP server on port 8000
3. Running queries and collecting metrics
4. Viewing metrics in both Python dict and Prometheus format
5. Monitoring query latency and index statistics
"""

import numpy as np
import time
import threading
import requests
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from rust_annie import AnnIndex, Distance, PyMetricsCollector
    print("✓ Successfully imported rust_annie with monitoring support")
except ImportError as e:
    print(f"✗ Failed to import rust_annie: {e}")
    sys.exit(1)

def test_metrics_collection():
    """Test basic metrics collection functionality."""
    print("\n=== Testing Metrics Collection ===")
    
    # Create an index with monitoring enabled
    index = AnnIndex(4, Distance.EUCLIDEAN)
    
    # Enable metrics collection with HTTP server on port 8000
    print("1. Enabling metrics on port 8000...")
    index.enable_metrics(8000)
    
    # Add some test data
    print("2. Adding test data...")
    data = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0, 6.0],
        [4.0, 5.0, 6.0, 7.0],
        [5.0, 6.0, 7.0, 8.0]
    ], dtype=np.float32)
    
    ids = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    index.add(data, ids)
    
    # Run some queries to generate metrics
    print("3. Running queries to generate metrics...")
    query = np.array([2.5, 3.5, 4.5, 5.5], dtype=np.float32)
    
    for i in range(10):
        labels, distances = index.search(query, k=3)
        time.sleep(0.01)  # Small delay to generate measurable latency
        
        # Add some variation to the query
        query_variant = query + np.random.normal(0, 0.1, size=4).astype(np.float32)
        labels2, distances2 = index.search(query_variant, k=2)
        time.sleep(0.005)
    
    # Get metrics
    print("4. Getting metrics...")
    metrics = index.get_metrics()
    
    if metrics:
        print(f"   Query count: {metrics['query_count']}")
        print(f"   Average latency: {metrics['avg_query_latency_us']:.2f} μs")
        print(f"   Index size: {metrics['index_size']}")
        print(f"   Dimensions: {metrics['dimensions']}")
        print(f"   Distance metric: {metrics['distance_metric']}")
        print(f"   Uptime: {metrics['uptime_seconds']} seconds")
    else:
        print("   No metrics available")
    
    return True

def test_prometheus_endpoint():
    """Test the Prometheus metrics endpoint."""
    print("\n=== Testing Prometheus Endpoint ===")
    
    # Give the server a moment to start
    time.sleep(1)
    
    try:
        # Test the metrics endpoint
        print("1. Testing /metrics endpoint...")
        response = requests.get("http://localhost:8000/metrics", timeout=5)
        
        if response.status_code == 200:
            print("   ✓ Metrics endpoint responding")
            metrics_text = response.text
            
            # Check for expected metrics
            expected_metrics = [
                "annie_queries_total",
                "annie_query_latency_microseconds_avg",
                "annie_index_size",
                "annie_dimensions",
                "annie_uptime_seconds"
            ]
            
            missing_metrics = []
            for metric in expected_metrics:
                if metric not in metrics_text:
                    missing_metrics.append(metric)
            
            if missing_metrics:
                print(f"   ✗ Missing metrics: {missing_metrics}")
                return False
            else:
                print("   ✓ All expected metrics found")
                
                # Show a sample of the metrics
                print("\n   Sample metrics output:")
                lines = metrics_text.split('\n')
                for line in lines[:10]:  # Show first 10 lines
                    if line.strip() and not line.startswith('#'):
                        print(f"   {line}")
                
                return True
        else:
            print(f"   ✗ Metrics endpoint returned status {response.status_code}")
            return False
            
    except requests.RequestException as e:
        print(f"   ✗ Failed to connect to metrics endpoint: {e}")
        return False

def test_health_endpoint():
    """Test the health check endpoint."""
    print("\n=== Testing Health Endpoint ===")
    
    try:
        print("1. Testing /health endpoint...")
        response = requests.get("http://localhost:8000/health", timeout=5)
        
        if response.status_code == 200 and response.text.strip() == "OK":
            print("   ✓ Health endpoint responding correctly")
            return True
        else:
            print(f"   ✗ Health endpoint returned: {response.status_code} - {response.text}")
            return False
            
    except requests.RequestException as e:
        print(f"   ✗ Failed to connect to health endpoint: {e}")
        return False

def test_standalone_metrics_collector():
    """Test the standalone PyMetricsCollector."""
    print("\n=== Testing Standalone Metrics Collector ===")
    
    # Create a standalone metrics collector
    collector = PyMetricsCollector()
    
    # Enable metrics on a different port
    print("1. Starting standalone metrics collector on port 8001...")
    collector.enable_metrics(8001)
    
    # Get initial metrics
    metrics = collector.get_metrics()
    print(f"2. Initial metrics: {metrics}")
    
    # Get Prometheus format
    prometheus_metrics = collector.get_prometheus_metrics()
    print("3. Prometheus format metrics:")
    print(prometheus_metrics[:200] + "..." if len(prometheus_metrics) > 200 else prometheus_metrics)
    
    return True

def test_concurrent_queries():
    """Test metrics collection under concurrent load."""
    print("\n=== Testing Concurrent Query Metrics ===")
    
    # Create an index
    index = AnnIndex(8, Distance.COSINE)
    index.enable_metrics(8002)
    
    # Add data
    data = np.random.random((50, 8)).astype(np.float32)
    ids = np.arange(50, dtype=np.int64)
    index.add(data, ids)
    
    # Function to run queries in a thread
    def run_queries(thread_id, num_queries):
        for i in range(num_queries):
            query = np.random.random(8).astype(np.float32)
            labels, distances = index.search(query, k=5)
            time.sleep(0.001)  # Small delay
    
    # Start multiple threads
    print("1. Starting concurrent queries...")
    threads = []
    for i in range(3):
        t = threading.Thread(target=run_queries, args=(i, 20))
        threads.append(t)
        t.start()
    
    # Wait for all threads to complete
    for t in threads:
        t.join()
    
    # Check metrics
    metrics = index.get_metrics()
    print(f"2. Total queries processed: {metrics['query_count']}")
    print(f"   Average latency: {metrics['avg_query_latency_us']:.2f} μs")
    
    # Verify we processed the expected number of queries
    expected_queries = 3 * 20  # 3 threads * 20 queries each
    if metrics['query_count'] == expected_queries:
        print("   ✓ Correct number of queries recorded")
        return True
    else:
        print(f"   ✗ Expected {expected_queries} queries, got {metrics['query_count']}")
        return False

def main():
    """Run all tests."""
    print("Testing Streaming K-NN Recall & Latency Monitoring")
    print("=" * 60)
    
    results = []
    
    try:
        # Test 1: Basic metrics collection
        results.append(test_metrics_collection())
        
        # Test 2: Prometheus endpoint
        results.append(test_prometheus_endpoint())
        
        # Test 3: Health endpoint
        results.append(test_health_endpoint())
        
        # Test 4: Standalone metrics collector
        results.append(test_standalone_metrics_collector())
        
        # Test 5: Concurrent queries
        results.append(test_concurrent_queries())
        
        # Summary
        print("\n" + "=" * 60)
        total_tests = len(results)
        passed_tests = sum(results)
        
        print(f"Tests Summary: {passed_tests}/{total_tests} passed")
        
        if passed_tests == total_tests:
            print("✓ All tests passed! The monitoring system is working correctly.")
            print("\nUsage examples:")
            print("  1. Enable metrics: index.enable_metrics(8000)")
            print("  2. View metrics: curl http://localhost:8000/metrics")
            print("  3. Health check: curl http://localhost:8000/health")
            print("  4. Get metrics in Python: index.get_metrics()")
            return 0
        else:
            print("✗ Some tests failed. Please check the output above.")
            return 1
            
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
