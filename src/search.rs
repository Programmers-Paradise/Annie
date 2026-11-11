//! Shared helpers for exact search: distance evaluation, top-k selection and
//! input validation. Used by both the brute-force and HNSW indexes so that
//! every code path reports identical distances.

use std::cmp::Ordering;

use ndarray::ArrayView2;
use pyo3::prelude::*;

use crate::errors::RustAnnError;
use crate::metrics::{self, Distance};

/// ID used to pad batch results when fewer than `k` neighbours exist.
pub const PAD_ID: i64 = -1;

/// Number of independent accumulators in the distance kernels. Splitting the
/// sum lets the compiler keep several SIMD lanes busy (floating-point addition
/// is not associative, so it will not reorder a single running sum itself).
const LANES: usize = 8;

/// Squared Euclidean distance.
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; LANES];
    let (ca, cb) = (a.chunks_exact(LANES), b.chunks_exact(LANES));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..LANES {
            let d = x[i] - y[i];
            acc[i] += d * d;
        }
    }
    let mut tail = 0.0;
    for (x, y) in ra.iter().zip(rb) {
        let d = x - y;
        tail += d * d;
    }
    acc.iter().sum::<f32>() + tail
}

/// Dot product.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; LANES];
    let (ca, cb) = (a.chunks_exact(LANES), b.chunks_exact(LANES));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..LANES {
            acc[i] += x[i] * y[i];
        }
    }
    let tail: f32 = ra.iter().zip(rb).map(|(x, y)| x * y).sum();
    acc.iter().sum::<f32>() + tail
}

/// Manhattan (L1) distance.
#[inline]
pub fn l1(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; LANES];
    let (ca, cb) = (a.chunks_exact(LANES), b.chunks_exact(LANES));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..LANES {
            acc[i] += (x[i] - y[i]).abs();
        }
    }
    let tail: f32 = ra.iter().zip(rb).map(|(x, y)| (x - y).abs()).sum();
    acc.iter().sum::<f32>() + tail
}

/// Ranking score for a built-in metric: monotone in the true distance, and
/// cheaper to compute (Euclidean is ranked by its square). Convert the final
/// results with [`score_to_distance`].
#[inline]
pub fn builtin_score(metric: &Distance, q: &[f32], q_norm: f32, v: &[f32], v_norm: f32) -> f32 {
    match metric {
        Distance::Euclidean() => l2_sq(q, v),
        _ => builtin_distance(metric, q, q_norm, v, v_norm),
    }
}

/// Convert a [`builtin_score`] into the distance reported to users.
#[inline]
pub fn score_to_distance(metric: &Distance, score: f32) -> f32 {
    match metric {
        Distance::Euclidean() => score.max(0.0).sqrt(),
        _ => score,
    }
}

/// Distance between `q` and `v` for a built-in metric.
///
/// `q_norm` / `v_norm` are the (non-squared) L2 norms, precomputed so that
/// cosine distance costs a single dot product per candidate.
#[inline]
pub fn builtin_distance(metric: &Distance, q: &[f32], q_norm: f32, v: &[f32], v_norm: f32) -> f32 {
    match metric {
        Distance::Euclidean() => l2_sq(q, v).sqrt(),
        Distance::Cosine() => {
            if q_norm == 0.0 || v_norm == 0.0 {
                1.0
            } else {
                (1.0 - dot(q, v) / (q_norm * v_norm)).clamp(0.0, 2.0)
            }
        }
        Distance::Manhattan() => l1(q, v),
        Distance::Chebyshev() => metrics::chebyshev(q, v),
        Distance::Minkowski(p) => metrics::minkowski(q, v, *p),
        Distance::Hamming() => metrics::hamming(q, v),
        Distance::Jaccard() => metrics::jaccard(q, v),
        Distance::Angular() => metrics::angular(q, v),
        Distance::Canberra() => metrics::canberra(q, v),
        Distance::Custom(_) => f32::NAN,
    }
}

/// Running top-k buffer: accepts candidates in any order, keeps memory at
/// O(k) by periodically discarding everything beyond the k best.
pub struct TopK {
    k: usize,
    buf: Vec<(i64, f32)>,
    /// Score of the k-th best candidate after the last shrink; candidates
    /// scoring worse can be skipped (NaN never beats it, ties are kept).
    bound: f32,
}

impl TopK {
    /// `k` must be at least 1. Memory is allocated lazily, so a `k` far larger
    /// than the number of candidates costs nothing.
    pub fn new(k: usize) -> Self {
        let k = k.max(1);
        TopK {
            k,
            buf: Vec::with_capacity(k.saturating_mul(2).clamp(64, 4096)),
            bound: f32::INFINITY,
        }
    }

    #[inline]
    pub fn push(&mut self, id: i64, score: f32) {
        if score > self.bound {
            return;
        }
        self.buf.push((id, score));
        if self.buf.len() >= self.k.max(32).saturating_mul(2) {
            self.shrink();
        }
    }

    fn shrink(&mut self) {
        if self.buf.len() > self.k {
            self.buf.select_nth_unstable_by(self.k - 1, cmp_candidates);
            self.buf.truncate(self.k);
            let kth = self.buf[self.k - 1].1;
            if !kth.is_nan() {
                self.bound = kth;
            }
        }
    }

    pub fn merge(mut self, other: TopK) -> TopK {
        for (id, s) in other.buf {
            self.push(id, s);
        }
        self
    }

    pub fn into_sorted(self) -> (Vec<i64>, Vec<f32>) {
        top_k(self.buf, self.k)
    }
}

#[inline]
pub fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Order by distance (NaN last), breaking ties by ID for deterministic output.
#[inline]
pub(crate) fn cmp_candidates(a: &(i64, f32), b: &(i64, f32)) -> Ordering {
    match (a.1.is_nan(), b.1.is_nan()) {
        (true, true) => a.0.cmp(&b.0),
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0)),
    }
}

/// Select the `k` nearest candidates, sorted by increasing distance.
pub fn top_k(mut candidates: Vec<(i64, f32)>, k: usize) -> (Vec<i64>, Vec<f32>) {
    let k = k.min(candidates.len());
    if k == 0 {
        return (Vec::new(), Vec::new());
    }
    if k < candidates.len() {
        candidates.select_nth_unstable_by(k - 1, cmp_candidates);
        candidates.truncate(k);
    }
    candidates.sort_unstable_by(cmp_candidates);
    candidates.into_iter().unzip()
}

/// Largest `n * k` a batch search will return (about 4.3e9 entries, 48 GiB of ids and
/// distances). Some systems (e.g. macOS) overcommit memory, so `try_reserve`
/// alone does not stop an absurd request from getting the process killed.
pub const MAX_BATCH_RESULTS: usize = u32::MAX as usize;

/// Allocate flat `(n * k)` output buffers for a batch search, raising
/// `MemoryError` instead of aborting the process when `n * k` is too large.
pub fn batch_buffers(n: usize, k: usize) -> PyResult<(Vec<i64>, Vec<f32>)> {
    let too_big = || {
        pyo3::exceptions::PyMemoryError::new_err(format!(
            "Cannot allocate results for {} queries with k={}; use a smaller k",
            n, k
        ))
    };
    let total = n
        .checked_mul(k)
        .filter(|&t| t <= MAX_BATCH_RESULTS)
        .ok_or_else(too_big)?;
    let mut ids = Vec::new();
    let mut dists = Vec::new();
    ids.try_reserve_exact(total).map_err(|_| too_big())?;
    dists.try_reserve_exact(total).map_err(|_| too_big())?;
    Ok((ids, dists))
}

/// Append `(ids, dists)` to flat batch buffers, padding to exactly `k` columns.
pub fn push_padded(
    all_ids: &mut Vec<i64>,
    all_dists: &mut Vec<f32>,
    ids: Vec<i64>,
    dists: Vec<f32>,
    k: usize,
) {
    let n = ids.len();
    all_ids.extend(ids);
    all_dists.extend(dists);
    for _ in n..k {
        all_ids.push(PAD_ID);
        all_dists.push(f32::INFINITY);
    }
}

pub fn check_k(k: usize) -> PyResult<()> {
    if k == 0 {
        return Err(RustAnnError::py_err("Invalid k", "k must be >= 1"));
    }
    Ok(())
}

pub fn check_query(q: &[f32], dim: usize) -> PyResult<()> {
    if q.len() != dim {
        return Err(RustAnnError::py_err(
            "Dimension Error",
            format!("Expected query of dimension {}, got {}", dim, q.len()),
        ));
    }
    if q.iter().any(|x| !x.is_finite()) {
        return Err(RustAnnError::py_err(
            "Invalid Input",
            "Query contains NaN or infinite values",
        ));
    }
    Ok(())
}

pub fn check_queries(queries: &ArrayView2<f32>, dim: usize) -> PyResult<()> {
    if queries.ncols() != dim {
        return Err(RustAnnError::py_err(
            "Dimension Error",
            format!(
                "Expected queries of shape (n, {}), got (n, {})",
                dim,
                queries.ncols()
            ),
        ));
    }
    if queries.iter().any(|x| !x.is_finite()) {
        return Err(RustAnnError::py_err(
            "Invalid Input",
            "Queries contain NaN or infinite values",
        ));
    }
    Ok(())
}

/// Validate a batch of vectors and IDs before insertion.
///
/// Checks shape, finiteness, non-negative IDs, duplicates within the batch,
/// and duplicates against `exists`.
pub fn check_insert(
    data: &ArrayView2<f32>,
    ids: &[i64],
    dim: usize,
    exists: impl Fn(i64) -> bool,
) -> PyResult<()> {
    if data.nrows() != ids.len() {
        return Err(RustAnnError::py_err(
            "Input Mismatch",
            format!(
                "data has {} rows but ids has {} entries",
                data.nrows(),
                ids.len()
            ),
        ));
    }
    if data.ncols() != dim {
        return Err(RustAnnError::py_err(
            "Dimension Error",
            format!(
                "Expected data of shape (n, {}), got (n, {})",
                dim,
                data.ncols()
            ),
        ));
    }
    if ids.iter().any(|&id| id < 0) {
        return Err(RustAnnError::py_err(
            "Invalid ID",
            "IDs must be non-negative",
        ));
    }
    if data.iter().any(|x| !x.is_finite()) {
        return Err(RustAnnError::py_err(
            "Invalid Input",
            "data contains NaN or infinite values",
        ));
    }
    let mut seen = std::collections::HashSet::with_capacity(ids.len());
    let mut duplicates: Vec<i64> = ids
        .iter()
        .copied()
        .filter(|&id| !seen.insert(id) || exists(id))
        .collect();
    if !duplicates.is_empty() {
        duplicates.sort_unstable();
        duplicates.dedup();
        duplicates.truncate(20);
        return Err(RustAnnError::py_err(
            "Duplicate IDs",
            format!("IDs already present or repeated in batch: {:?}", duplicates),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_sorts_and_truncates() {
        let c = vec![(1, 3.0), (2, 1.0), (3, 2.0), (4, f32::NAN)];
        let (ids, d) = top_k(c, 3);
        assert_eq!(ids, vec![2, 3, 1]);
        assert_eq!(d, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn top_k_ties_broken_by_id() {
        let c = vec![(5, 1.0), (2, 1.0), (9, 1.0)];
        assert_eq!(top_k(c, 2).0, vec![2, 5]);
    }

    #[test]
    fn top_k_handles_k_larger_than_n() {
        let (ids, _) = top_k(vec![(1, 0.5)], 10);
        assert_eq!(ids, vec![1]);
    }

    #[test]
    fn euclidean_is_exact_for_identical_vectors() {
        let v = [0.1_f32, 0.2, 0.3];
        let d = builtin_distance(&Distance::Euclidean(), &v, l2_norm(&v), &v, l2_norm(&v));
        assert_eq!(d, 0.0);
    }

    #[test]
    fn padding_fills_to_k() {
        let (mut i, mut d) = (vec![], vec![]);
        push_padded(&mut i, &mut d, vec![7], vec![0.1], 3);
        assert_eq!(i, vec![7, PAD_ID, PAD_ID]);
        assert!(d[1].is_infinite());
    }
}
