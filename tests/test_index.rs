use rust_annie::index::AnnIndex;
use rust_annie::metrics::Distance;
use pyo3::Python;
use numpy::PyArrayMethods;

#[test]
fn test_brute_backend() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let mut index = AnnIndex::new(3, Distance::Euclidean()).unwrap();

        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        let ids = vec![10, 20, 30];

        // Convert to numpy arrays via PyO3
        let np_data = numpy::PyArray2::from_vec2(py, &data).unwrap();
        let np_ids = numpy::PyArray1::from_slice(py, &ids);

        index.add(py, np_data.readonly(), np_ids.readonly()).unwrap();

        let query = numpy::PyArray1::from_slice(py, &[1.0, 2.0, 3.0]);
        let (res_ids, res_dists) = index.search(py, query.readonly(), 2).unwrap();

        let ids: Vec<usize> = res_ids.extract(py).unwrap();
        let dists: Vec<f32> = res_dists.extract(py).unwrap();

        println!("IDs: {:?}", ids);
        println!("Dists: {:?}", dists);
    });
}

#[test]
fn test_hnsw_backend() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let mut index = AnnIndex::new(3, Distance::Euclidean()).unwrap();

        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        let ids = vec![10, 20, 30];

        let np_data = numpy::PyArray2::from_vec2(py, &data).unwrap();
        let np_ids = numpy::PyArray1::from_slice(py, &ids);

        index.add(py, np_data.readonly(), np_ids.readonly()).unwrap();

        let query = numpy::PyArray1::from_slice(py, &[4.0, 5.0, 6.0]);
        let (res_ids, res_dists) = index.search(py, query.readonly(), 2).unwrap();

        let ids: Vec<usize> = res_ids.extract(py).unwrap();
        let dists: Vec<f32> = res_dists.extract(py).unwrap();

        println!("IDs: {:?}", ids);
        println!("Dists: {:?}", dists);
    });
}
