"""
Demonstration of the Annie monitoring system.
This script starts an index with metrics enabled and keeps it running
so you can test the endpoints manually.
"""

import numpy as np
import time
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rust_annie import AnnIndex, Distance

def main():
    print("Annie Monitoring System Demo")
    print("=" * 30)
    
    # Create an index with monitoring
    print("1. Creating index with Euclidean distance...")
    index = AnnIndex(128, Distance.EUCLIDEAN)
    
    # Enable metrics on port 8000
    print("2. Enabling metrics on port 8000...")
    index.enable_metrics(8000)
    
    # Add some sample data
    print("3. Adding sample data...")
    data = np.random.random((1000, 128)).astype(np.float32)
    ids = np.arange(1000, dtype=np.int64)
    index.add(data, ids)
    
    print("4. Running sample queries...")
    # Run some queries to generate metrics
    for i in range(50):
        query = np.random.random(128).astype(np.float32)
        labels, distances = index.search(query, k=10)
        if i % 10 == 0:
            print(f"   Completed {i + 1} queries...")
    
    # Show current metrics
    print("5. Current metrics:")
    metrics = index.get_metrics()
    for key, value in metrics.items():
        if key == 'recall_estimates':
            continue  # Skip empty recall estimates
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 50)
    print("Metrics server is running on port 8000!")
    print("Try these commands in another terminal:")
    print("  curl http://localhost:8000/metrics")
    print("  curl http://localhost:8000/health")
    print("\nPress Ctrl+C to stop the server...")
    
    try:
        # Keep the server running
        while True:
            # Run a few more queries periodically
            for _ in range(5):
                query = np.random.random(128).astype(np.float32)
                labels, distances = index.search(query, k=5)
            
            time.sleep(10)  # Wait 10 seconds between batches
            
            # Update metrics display
            metrics = index.get_metrics()
            print(f"\rQueries: {metrics['query_count']}, Avg Latency: {metrics['avg_query_latency_us']:.2f}μs", end='', flush=True)
            
    except KeyboardInterrupt:
        print("\n\nStopping server...")
        return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nStopping server...")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        sys.exit(1)
