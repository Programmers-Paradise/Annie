use std::sync::{Arc, RwLock};
use rust_annie::concurrency::ThreadSafeAnnIndex;
use rust_annie::index::AnnIndex;
use rust_annie::metrics::Distance;
use std::thread;
use numpy::PyArrayMethods;
use ndarray::Array2;
use numpy::PyArray2;
use pyo3::Python;

#[test]
fn stress_concurrent_search_and_add() {
    pyo3::prepare_freethreaded_python();
    let lock = Arc::new(RwLock::new(AnnIndex::new(8, Distance::Euclidean()).unwrap()));
    let wrapped = Arc::new(ThreadSafeAnnIndex::from_arc(lock.clone()));
    let mut handles = vec![];
    for _ in 0..32 {
        let w = wrapped.clone();
        handles.push(thread::spawn(move || {
            Python::with_gil(|py| {
                let arr = vec![[1.0_f32; 8], [2.0; 8]];
                let arr = PyArray2::from_owned_array(py, Array2::from_shape_vec((2, 8), arr.concat()).unwrap());
                let _ = w.search_batch(py, arr.readonly(), 1);
            });
        }));
    }
    for h in handles { let _ = h.join(); }
    // Add data concurrently
    let mut add_handles = vec![];
    for i in 0..16 {
        let lock = lock.clone();
        add_handles.push(thread::spawn(move || {
            let mut index = lock.write().unwrap();
            Python::with_gil(|py| {
                let arr = vec![vec![i as f32; 8]];
                let ids = vec![i];
                let np_data = PyArray2::from_vec2(py, &arr).unwrap();
                let np_ids = numpy::PyArray1::from_slice(py, &ids);
                let _ = index.add(py, np_data.readonly(), np_ids.readonly());
            });
        }));
    }
    for h in add_handles { let _ = h.join(); }
}
