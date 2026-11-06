//! Tests of the pure-Rust API. Python-level behaviour (GIL handling, NumPy
//! conversion, thread safety) is covered by the pytest suite in tests/python.

use ndarray::{array, Array2};
use rust_annie::filters::Filter;
use rust_annie::hnsw_index::{HnswConfig, HnswIndex};
use rust_annie::index::AnnIndex;
use rust_annie::metrics::Distance;
use rust_annie::path_validation::validate_path_secure;
use rust_annie::storage::{detect_kind, IndexKind};

fn grid() -> (Array2<f32>, Vec<i64>) {
    let data = Array2::from_shape_fn((100, 2), |(i, j)| {
        if j == 0 {
            (i % 10) as f32
        } else {
            (i / 10) as f32
        }
    });
    (data, (0..100).map(|i| i * 3).collect())
}

#[test]
fn brute_force_exact_results() {
    let (data, ids) = grid();
    let mut idx = AnnIndex::create(2, Distance::Euclidean()).unwrap();
    idx.add_rows(data.view(), &ids).unwrap();
    let (found, dists) = idx.search_vec(&[0.0, 0.0], 3, None).unwrap();
    assert_eq!(found, vec![0, 3, 30]); // (0,0), (1,0), (0,1); tie broken by id
    assert_eq!(dists, vec![0.0, 1.0, 1.0]);
}

#[test]
fn brute_force_rejects_bad_input() {
    assert!(AnnIndex::create(0, Distance::Euclidean()).is_err());
    let mut idx = AnnIndex::create(2, Distance::Euclidean()).unwrap();
    assert!(idx.add_rows(array![[1.0, 2.0, 3.0]].view(), &[0]).is_err());
    assert!(idx.add_rows(array![[1.0, 2.0]].view(), &[-5]).is_err());
    assert!(idx.add_rows(array![[f32::NAN, 2.0]].view(), &[0]).is_err());
    assert!(idx
        .add_rows(array![[1.0, 2.0], [3.0, 4.0]].view(), &[1, 1])
        .is_err());
    assert_eq!(idx.size(), 0);
    idx.add_rows(array![[1.0, 2.0]].view(), &[1]).unwrap();
    assert!(idx.search_vec(&[1.0], 1, None).is_err());
    assert!(idx.search_vec(&[1.0, 2.0], 0, None).is_err());
}

#[test]
fn brute_force_remove_update_compact() {
    let (data, ids) = grid();
    let mut idx = AnnIndex::create(2, Distance::Manhattan()).unwrap();
    idx.add_rows(data.view(), &ids).unwrap();
    idx.remove_ids(&[0]).unwrap();
    assert!(idx.remove_ids(&[0]).is_err());
    assert_eq!(idx.search_vec(&[0.0, 0.0], 1, None).unwrap().0, vec![3]);
    idx.update_vector(297, &[0.0, 0.0]).unwrap();
    assert_eq!(idx.search_vec(&[0.0, 0.0], 1, None).unwrap().0, vec![297]);
    idx.compact_entries();
    assert_eq!(idx.size(), 99);
}

#[test]
fn filters_restrict_results() {
    let (data, ids) = grid();
    let mut idx = AnnIndex::create(2, Distance::Euclidean()).unwrap();
    idx.add_rows(data.view(), &ids).unwrap();
    let f = Filter::and(vec![
        Filter::id_range(100, 200),
        Filter::not(Filter::id_set(vec![102])),
    ]);
    let (found, _) = idx.search_vec(&[0.0, 0.0], 100, Some(&f)).unwrap();
    assert!(found.iter().all(|&i| (100..=200).contains(&i) && i != 102));
    assert_eq!(found.len(), 32); // 33 multiples of 3 in [100, 200], minus 102
}

#[test]
fn batch_search_pads_rows() {
    let mut idx = AnnIndex::create(1, Distance::Euclidean()).unwrap();
    idx.add_rows(array![[0.0], [1.0]].view(), &[5, 6]).unwrap();
    let (ids, dists) = idx
        .search_batch_vec(array![[0.0], [1.0]].view(), 3, None)
        .unwrap();
    assert_eq!(ids, vec![5, 6, -1, 6, 5, -1]);
    assert!(dists[2].is_infinite() && dists[5].is_infinite());
}

#[test]
fn hnsw_maps_ids_and_reports_distances() {
    let (data, ids) = grid();
    let mut idx = HnswIndex::new(2, Distance::Euclidean(), HnswConfig::default()).unwrap();
    idx.add_rows(data.view(), &ids).unwrap();
    let (found, dists) = idx.search_vec(&[9.0, 9.0], 1, None, None).unwrap();
    assert_eq!(found, vec![297]);
    assert_eq!(dists, vec![0.0]);
    idx.remove_ids(&[297]).unwrap();
    assert_ne!(
        idx.search_vec(&[9.0, 9.0], 1, Some(200), None).unwrap().0,
        vec![297]
    );
    idx.validate().unwrap();
}

#[test]
fn hnsw_rejects_bad_config_without_exiting() {
    let bad = HnswConfig {
        m: 1000,
        ..HnswConfig::default()
    };
    assert!(HnswIndex::new(4, Distance::Euclidean(), bad).is_err());
    assert!(HnswIndex::new(4, Distance::Hamming(), HnswConfig::default()).is_err());
}

#[test]
fn persistence_roundtrip() {
    let dir = std::env::temp_dir().join(format!("rust_annie_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (data, ids) = grid();

    let mut brute = AnnIndex::create(2, Distance::Cosine()).unwrap();
    brute.add_rows(data.view(), &ids).unwrap();
    let p = dir.join("brute.bin");
    brute.save_to(&p).unwrap();
    assert_eq!(detect_kind(&p).unwrap(), IndexKind::BruteForce);
    let loaded = AnnIndex::load_from(&p).unwrap();
    assert_eq!(loaded.size(), 100);
    assert_eq!(
        loaded.search_vec(&[3.0, 4.0], 5, None).unwrap(),
        brute.search_vec(&[3.0, 4.0], 5, None).unwrap()
    );

    let mut hnsw = HnswIndex::new(2, Distance::Euclidean(), HnswConfig::default()).unwrap();
    hnsw.add_rows(data.view(), &ids).unwrap();
    let p = dir.join("hnsw.bin");
    hnsw.save_to(&p).unwrap();
    assert_eq!(detect_kind(&p).unwrap(), IndexKind::Hnsw);
    assert!(AnnIndex::load_from(&p).is_err());
    let loaded = HnswIndex::load_from(&p).unwrap();
    assert_eq!(loaded.size(), 100);
    assert_eq!(
        loaded.search_vec(&[2.0, 2.0], 1, None, None).unwrap().0,
        vec![66]
    );

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn path_validation() {
    assert!(validate_path_secure("index.bin").is_ok());
    assert!(validate_path_secure(std::env::temp_dir().join("x.bin").to_str().unwrap()).is_ok());
    assert!(validate_path_secure("").is_err());
    assert!(validate_path_secure("a\0b").is_err());
}
