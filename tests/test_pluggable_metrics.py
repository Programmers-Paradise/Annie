"""
Test script to verify the pluggable distance metric registry works.
"""

import numpy as np
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from rust_annie import AnnIndex, Distance, register_metric, list_metrics
    print("✓ Successfully imported rust_annie")
except ImportError as e:
    print(f"✗ Failed to import rust_annie: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic functionality of the pluggable distance metric registry."""
    print("\n=== Testing Basic Functionality ===")
    
    # Test 1: List built-in metrics
    print("1. Built-in metrics:")
    metrics = list_metrics()
    print(f"   Available metrics: {sorted(metrics)}")
    
    # Test 2: Create Distance objects
    print("\n2. Creating Distance objects:")
    euclidean = Distance("euclidean")
    print(f"   Euclidean distance name: {euclidean.name()}")
    
    custom = Distance.custom("my_custom")
    print(f"   Custom distance name: {custom.name()}")
    
    # Test 3: Register a custom metric
    print("\n3. Registering custom L1.5 metric:")
    
    def l1_5_distance(a, b):
        """L1.5 norm distance"""
        return np.sum(np.abs(np.array(a) - np.array(b)) ** 1.5) ** (1.0 / 1.5)
    
    register_metric("l1_5", l1_5_distance)
    print("   ✓ Registered L1.5 metric")
    
    # Test 4: Verify the metric is listed
    print("\n4. Updated metrics list:")
    metrics = list_metrics()
    print(f"   Available metrics: {sorted(metrics)}")
    assert "l1_5" in metrics, "L1.5 metric should be in the list"
    
    # Test 5: Create index with custom metric
    print("\n5. Creating index with custom metric:")
    index = AnnIndex.new_with_metric(2, "l1_5")
    print("   ✓ Created index with L1.5 metric")
    
    # Test 6: Add data and search
    print("\n6. Testing search with custom metric:")
    data = np.array([
        [1.0, 2.0],
        [3.0, 1.0],
        [2.0, 3.0]
    ], dtype=np.float32)
    
    ids = np.array([0, 1, 2], dtype=np.int64)
    index.add(data, ids)
    
    query = np.array([2.0, 2.0], dtype=np.float32)
    labels, distances = index.search(query, k=2)
    
    print(f"   Query: {query}")
    print(f"   Top 2 results: labels={labels}, distances={distances}")
    
    # Test 7: Compare with built-in metrics
    print("\n7. Comparing with built-in Euclidean:")
    index_euclidean = AnnIndex(2, Distance.EUCLIDEAN)
    index_euclidean.add(data, ids)
    
    labels_euclidean, distances_euclidean = index_euclidean.search(query, k=2)
    print(f"   Euclidean results: labels={labels_euclidean}, distances={distances_euclidean}")
    
    return True

def test_advanced_functionality():
    """Test advanced functionality."""
    print("\n=== Testing Advanced Functionality ===")
    
    # Test 8: Register a more complex metric (weighted Euclidean)
    print("8. Registering weighted Euclidean metric:")
    
    weights = np.array([1.0, 2.0])  # Give more weight to second dimension
    
    def weighted_euclidean(a, b):
        """Weighted Euclidean distance"""
        diff = np.array(a) - np.array(b)
        return np.sqrt(np.sum(weights * (diff ** 2)))
    
    register_metric("weighted_euclidean", weighted_euclidean)
    print("   ✓ Registered weighted Euclidean metric")
    
    # Test 9: Use the weighted metric
    print("\n9. Testing weighted Euclidean:")
    index_weighted = AnnIndex.new_with_metric(2, "weighted_euclidean")
    
    data = np.array([
        [0.0, 0.0],
        [1.0, 0.0],  # Different in first dimension
        [0.0, 1.0],  # Different in second dimension (should be farther due to weight)
    ], dtype=np.float32)
    
    ids = np.array([0, 1, 2], dtype=np.int64)
    index_weighted.add(data, ids)
    
    query = np.array([0.0, 0.0], dtype=np.float32)
    labels, distances = index_weighted.search(query, k=3)
    
    print(f"   Query: {query}")
    print(f"   Results: labels={labels}, distances={distances}")
    print(f"   Point [1,0] distance: {distances[labels.tolist().index(1)]:.4f}")
    print(f"   Point [0,1] distance: {distances[labels.tolist().index(2)]:.4f}")
    print("   (Point [0,1] should be farther due to higher weight on second dimension)")
    
    return True

def main():
    """Run all tests."""
    print("Testing Pluggable Distance Metric Registry")
    print("=" * 50)
    
    try:
        test_basic_functionality()
        test_advanced_functionality()
        print("\n✓ All tests passed!")
        return 0
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
