"""
Example demonstrating the pluggable distance metric registry.
"""
import numpy as np
from rust_annie import AnnIndex, Distance, register_metric, list_metrics

def main():
    print("=== Pluggable Distance Metric Registry Demo ===\n")
    
    # 1. Show built-in metrics
    print("1. Built-in metrics:")
    metrics = list_metrics()
    for metric in sorted(metrics):
        print(f"   - {metric}")
    print()
    
    # 2. Register a custom L1.5 distance metric
    print("2. Registering custom L1.5 distance metric...")
    
    def l1_5_distance(a, b):
        """L1.5 norm distance (between Manhattan and Euclidean)"""
        a = np.asarray(a)
        b = np.asarray(b)
        if a.shape != b.shape:
            raise ValueError(f"Input arrays must have the same shape, got {a.shape} and {b.shape}")
        return np.sum(np.abs(a - b) ** 1.5) ** (1.0 / 1.5)

    
    register_metric("l1_5", l1_5_distance)
    print("   Registered 'l1_5' metric")
    
    # 3. Register a Mahalanobis distance metric
    print("3. Registering Mahalanobis distance metric...")
    
    # Example covariance matrix
    cov_matrix = np.array([[2.0, 0.3], [0.3, 1.0]], dtype=np.float32)
    cov_inv = np.linalg.inv(cov_matrix)
    
    def mahalanobis_distance(a, b):
        """Mahalanobis distance using predefined covariance matrix"""
        global cov_inv
        if 'cov_inv' not in globals():
            raise RuntimeError("cov_inv must be defined globally before using mahalanobis_distance.")
        a_arr = np.array(a)
        b_arr = np.array(b)
        if a_arr.shape != b_arr.shape:
            raise ValueError(f"Input vectors must have the same shape, got {a_arr.shape} and {b_arr.shape}")
        diff = a_arr - b_arr
        if diff.shape[0] != cov_inv.shape[0]:
            raise ValueError(f"Input vectors must have dimension {cov_inv.shape[0]}, got {diff.shape[0]}")
        return np.sqrt(np.dot(diff, np.dot(cov_inv, diff)))
    
    register_metric("mahalanobis", mahalanobis_distance)
    print("   Registered 'mahalanobis' metric")
    
    # 4. Show updated metrics list
    print("\n4. Updated metrics list:")
    metrics = list_metrics()
    for metric in sorted(metrics):
        print(f"   - {metric}")
    print()
    
    # 5. Create indices with different metrics
    print("5. Creating indices with different metrics...")
    
    # Sample data
    data = np.array([
        [1.0, 2.0],
        [3.0, 1.0],
        [2.0, 3.0],
        [4.0, 2.0],
        [1.5, 1.5]
    ], dtype=np.float32)
    
    ids = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    query = np.array([2.0, 2.0], dtype=np.float32)
    
    # Test different metrics
    metrics_to_test = ["euclidean", "manhattan", "l1_5", "mahalanobis"]
    
    for metric_name in metrics_to_test:
        print(f"\n   Testing {metric_name}:")
        
        # Create index
        if metric_name in ["euclidean", "manhattan"]:
            # Use built-in Distance enum
            distance = Distance(metric_name)
            index = AnnIndex(2, distance)
        else:
            # Use custom metric by name
            index = AnnIndex.new_with_metric(2, metric_name)
        
        # Add data
        index.add(data, ids)
        
        # Search
        labels, distances = index.search(query, k=3)
        
        print(f"     Top 3 nearest neighbors:")
        for i, (label, dist) in enumerate(zip(labels, distances)):
            point = data[label]
            print(f"       {i+1}. ID {label}: {point} (distance: {dist:.4f})")
    
    # 6. Demonstrate using Distance.custom() method
    print("\n6. Using Distance.custom() method:")
    
    # Create a custom distance using the Distance class
    custom_distance = Distance.custom("l1_5")
    index = AnnIndex(2, custom_distance)
    index.add(data, ids)
    
    labels, distances = index.search(query, k=1)
    print(f"   Closest point using Distance.custom('l1_5'): ID {labels[0]}, distance: {distances[0]:.4f}")
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    main()
