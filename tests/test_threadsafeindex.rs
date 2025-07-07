#[test]
fn test_search_batch_poisoned_lock() {
    use std::sync::{Arc, RwLock};
    use rust_annie::concurrency::ThreadSafeAnnIndex;
    use rust_annie::index::AnnIndex;
    use rust_annie::metrics::Distance;
    use pyo3::Python;
    use numpy::{PyArray2, IntoPyArray};

    // Poison the lock
    let lock = Arc::new(RwLock::new(AnnIndex::new(2, Distance::L2).unwrap()));
    {
        let poisoned = Arc::clone(&lock);
        let _ = std::thread::spawn(move || {
            let _ = poisoned.write().unwrap();
            panic!("Poisoning lock intentionally");
        }).join();
    }

    let wrapped = ThreadSafeAnnIndex { inner: lock };

    Python::with_gil(|py| {
        let arr = vec![[1.0_f32, 2.0], [3.0, 4.0]];
        let arr = PyArray2::from_owned_array(py, ndarray::Array2::from_shape_vec((2, 2), arr.concat()).unwrap());
        let result = wrapped.search_batch(py, arr.readonly(), 1);
        assert!(result.is_err());
        let msg = format!("{:?}", result.unwrap_err());
        assert!(msg.contains("Lock Error"));
    });
}
