use rust_annie::index::AnnIndex;
use rust_annie::metrics::Distance;
use numpy::PyArrayMethods;
use numpy::{PyArray2, PyArray1};
use pyo3::Python;

#[test]
fn test_security_invalid_input() {
    // Negative dimension
    let index = AnnIndex::new(0, Distance::Euclidean());
    assert!(index.is_err(), "Should not allow zero dimension");
    // Oversized input
    let index = AnnIndex::new(10000, Distance::Euclidean());
    assert!(index.is_err(), "Should not allow excessive dimension");
}

#[test]
fn test_security_boundary_checks() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let mut index = AnnIndex::new(3, Distance::Euclidean()).unwrap();
        let vecs = vec![vec![1.0, 2.0, 3.0]];
        let ids = vec![1];
        let np_data = PyArray2::from_vec2(py, &vecs).unwrap();
        let np_ids = PyArray1::from_slice(py, &ids);
        let _ = index.add(py, np_data.readonly(), np_ids.readonly());
        // Out-of-bounds search
        let query = PyArray1::from_slice(py, &[1.0, 2.0]); // wrong dimension
        let result = index.search(py, query.readonly(), 1, None);
        assert!(result.is_err(), "Should error on wrong query dimension");
    });
}

#[test]
fn test_security_dos_attempt() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let mut index = AnnIndex::new(3, Distance::Euclidean()).unwrap();
        // Use one more than the safe limit to ensure the guard triggers
        let big_vecs = vec![vec![1.0; 3]; 1_000_001];
        let big_ids = (0..1_000_001).collect::<Vec<_>>();
        let np_data = PyArray2::from_vec2(py, &big_vecs).unwrap();
        let np_ids = PyArray1::from_slice(py, &big_ids);
        let result = index.add(py, np_data.readonly(), np_ids.readonly());
        assert!(result.is_err(), "Should error on excessive allocation");
    });
}
