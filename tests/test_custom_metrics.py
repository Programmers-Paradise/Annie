"""
Test the pluggable distance metric registry functionality.
"""
import numpy as np
import pytest
from rust_annie import AnnIndex, Distance, register_metric, list_metrics

def test_list_builtin_metrics():
    """Test that built-in metrics are available."""
    metrics = list_metrics()
    assert "euclidean" in metrics
    assert "cosine" in metrics
    assert "manhattan" in metrics
    assert "chebyshev" in metrics

def test_custom_distance_creation():
    """Test creating Distance objects with custom names."""
    # Test built-in metrics
    euclidean = Distance("euclidean")
    assert euclidean.name() == "euclidean"
    
    cosine = Distance("cosine")
    assert cosine.name() == "cosine"
    
    # Test custom metric
    custom = Distance("my_custom_metric")
    assert custom.name() == "my_custom_metric"
    
    # Test using the custom class method
    custom2 = Distance.custom("another_metric")
    assert custom2.name() == "another_metric"

def test_register_custom_metric():
    """Test registering a custom distance metric."""
    
    # Define a custom distance function (L1.5 norm)
    def l1_5_distance(a, b):
        return np.sum(np.abs(np.array(a) - np.array(b)) ** 1.5) ** (1.0 / 1.5)
    
    # Register the custom metric
    register_metric("l1_5", l1_5_distance)
    
    # Verify it's in the list
    metrics = list_metrics()
    assert "l1_5" in metrics
    
    # Create an index using the custom metric
    index = AnnIndex.new_with_metric(2, "l1_5")
    
    # Add some test data
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ids = np.array([0, 1], dtype=np.int64)
    index.add(data, ids)
    
    # Test search
    query = np.array([1.5, 2.5], dtype=np.float32)
    labels, distances = index.search(query, k=1)
    
    assert len(labels) == 1
    assert len(distances) == 1
    assert labels[0] == 0  # Should find the first point as closest

def test_register_manhattan_clone():
    """Test registering a custom implementation of Manhattan distance."""
    
    def custom_manhattan(a, b):
        return np.sum(np.abs(np.array(a) - np.array(b)))
    
    register_metric("custom_manhattan", custom_manhattan)
    
    # Create indices with built-in and custom Manhattan
    index_builtin = AnnIndex(2, Distance.MANHATTAN)
    index_custom = AnnIndex.new_with_metric(2, "custom_manhattan")
    
    # Add the same data to both
    data = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    ids = np.array([0, 1, 2], dtype=np.int64)
    
    index_builtin.add(data, ids)
    index_custom.add(data, ids)
    
    # Test that both give similar results
    query = np.array([0.5, 0.5], dtype=np.float32)
    
    labels_builtin, distances_builtin = index_builtin.search(query, k=3)
    labels_custom, distances_custom = index_custom.search(query, k=3)
    
    # Results should be the same (or very close)
    assert np.array_equal(labels_builtin, labels_custom)
    assert np.allclose(distances_builtin, distances_custom)
    np.testing.assert_allclose(distances_builtin, distances_custom, rtol=1e-5)

def test_mahalanobis_distance():
    """Test implementing Mahalanobis distance as a custom metric."""
    
    # Example covariance matrix (must be positive definite)
    cov_matrix = np.array([[2.0, 0.5], [0.5, 1.0]], dtype=np.float32)
    cov_inv = np.linalg.inv(cov_matrix)
    
    def mahalanobis_distance(a, b):
        diff = np.array(a) - np.array(b)
        return np.sqrt(np.dot(diff, np.dot(cov_inv, diff)))
    
    register_metric("mahalanobis", mahalanobis_distance)
    
    # Create index with Mahalanobis distance
    index = AnnIndex.new_with_metric(2, "mahalanobis")
    
    # Add test data
    data = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    ids = np.array([0, 1, 2, 3], dtype=np.int64)
    index.add(data, ids)
    
    # Test search
    query = np.array([0.5, 0.5], dtype=np.float32)
    labels, distances = index.search(query, k=2)
    
    assert len(labels) == 2
    assert len(distances) == 2
    # Results should be meaningful (positive distances)
    assert all(d >= 0 for d in distances)

def test_error_cases():
    """Test error handling for invalid cases."""
    
    # Test that unregistered custom metric still works (falls back to Euclidean)
    index = AnnIndex.new_with_metric(2, "nonexistent_metric")
    
    # Should still work (with fallback)
    data = np.array([[0.0, 0.0]], dtype=np.float32)
    ids = np.array([0], dtype=np.int64)
    index.add(data, ids)
    
    query = np.array([1.0, 1.0], dtype=np.float32)
    labels, distances = index.search(query, k=1)
    
    assert len(labels) == 1
    assert len(distances) == 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
