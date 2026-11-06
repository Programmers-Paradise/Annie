//! Exact (brute-force) k-nearest-neighbour index.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::Instant;

use ndarray::{Array2, ArrayView2};
use numpy::{AllowTypeChange, PyArrayLike1, PyArrayLike2, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::distance_registry::get_distance_function_safe;
use crate::errors::RustAnnError;
use crate::filters::Filter;
use crate::metrics::Distance;
use crate::monitoring::MetricsCollector;
use crate::path_validation::validate_path_secure;
use crate::search::{self, builtin_distance, l2_norm, top_k};
use crate::storage::{self, BruteForceData};

/// Maximum supported vector dimensionality.
pub const MAX_DIM: usize = 65_536;

/// A stored vector: `(id, vector, l2_norm)`.
pub(crate) type Entry = (i64, Vec<f32>, f32);

/// Exact k-nearest-neighbour index using brute-force search.
///
/// Every query is compared against every stored vector, so results are exact.
/// Distance computations are parallelised with Rayon and the GIL is released
/// while searching with a built-in metric.
///
/// Parameters
/// ----------
/// dim : int
///     Dimensionality of the vectors.
/// metric : Distance, default Distance.EUCLIDEAN
///     Distance metric used for search.
#[pyclass(module = "rust_annie")]
pub struct AnnIndex {
    pub(crate) dim: usize,
    pub(crate) metric: Distance,
    /// Storage slots; `None` marks a removed entry awaiting compaction.
    pub(crate) entries: Vec<Option<Entry>>,
    /// Map from user ID to storage slot.
    pub(crate) slots: HashMap<i64, usize>,
    pub(crate) deleted_count: usize,
    pub(crate) max_deleted_ratio: f32,
    pub(crate) metrics: Option<Arc<MetricsCollector>>,
    pub(crate) version: Arc<AtomicU64>,
}

// ---------------------------------------------------------------------------
// Pure-Rust API (usable without the Python bindings)
// ---------------------------------------------------------------------------
impl AnnIndex {
    /// Create an empty index. Rust equivalent of `AnnIndex(dim, metric)`.
    pub fn create(dim: usize, metric: Distance) -> Result<Self, PyErr> {
        if dim == 0 {
            return Err(RustAnnError::py_err(
                "Invalid Dimension",
                "Dimension must be > 0",
            ));
        }
        if dim > MAX_DIM {
            return Err(RustAnnError::py_err(
                "Excessive Dimension",
                format!("Dimension {} exceeds maximum {}", dim, MAX_DIM),
            ));
        }
        if let Distance::Minkowski(p) = metric {
            if p.is_nan() || p <= 0.0 {
                return Err(RustAnnError::py_err("Minkowski Error", "`p` must be > 0"));
            }
        }
        Ok(AnnIndex {
            dim,
            metric,
            entries: Vec::new(),
            slots: HashMap::new(),
            deleted_count: 0,
            max_deleted_ratio: 0.2,
            metrics: None,
            version: Arc::new(AtomicU64::new(0)),
        })
    }

    fn bump_version(&self) {
        self.version.fetch_add(1, AtomicOrdering::Relaxed);
        if let Some(m) = &self.metrics {
            m.set_index_metadata(self.slots.len(), self.dim, &self.metric.name());
        }
    }

    /// Insert rows of `data` with the given IDs. Validates all input first,
    /// so a failed call leaves the index unchanged.
    pub fn add_rows(&mut self, data: ArrayView2<f32>, ids: &[i64]) -> PyResult<()> {
        search::check_insert(&data, ids, self.dim, |id| self.slots.contains_key(&id))?;
        self.entries.reserve(ids.len());
        for (row, &id) in data.rows().into_iter().zip(ids) {
            let v = row.to_vec();
            let norm = l2_norm(&v);
            self.slots.insert(id, self.entries.len());
            self.entries.push(Some((id, v, norm)));
        }
        self.bump_version();
        Ok(())
    }

    /// Remove entries by ID. Raises `KeyError` (and changes nothing) if any ID
    /// is not present.
    pub fn remove_ids(&mut self, ids: &[i64]) -> PyResult<()> {
        let missing: Vec<i64> = ids
            .iter()
            .copied()
            .filter(|id| !self.slots.contains_key(id))
            .collect();
        if !missing.is_empty() {
            return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "IDs not found in index: {:?}",
                missing
            )));
        }
        for id in ids {
            if let Some(slot) = self.slots.remove(id) {
                self.entries[slot] = None;
                self.deleted_count += 1;
            }
        }
        if self.should_compact() {
            self.compact_entries();
        }
        self.bump_version();
        Ok(())
    }

    /// Replace the vector stored under `id`.
    pub fn update_vector(&mut self, id: i64, vector: &[f32]) -> PyResult<()> {
        search::check_query(vector, self.dim)?;
        let slot = *self
            .slots
            .get(&id)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(format!("ID {} not found", id)))?;
        self.entries[slot] = Some((id, vector.to_vec(), l2_norm(vector)));
        self.bump_version();
        Ok(())
    }

    /// Drop removed slots and rebuild the ID map.
    pub fn compact_entries(&mut self) {
        self.entries.retain(|e| e.is_some());
        self.slots = self
            .entries
            .iter()
            .enumerate()
            .map(|(slot, e)| (e.as_ref().unwrap().0, slot))
            .collect();
        self.deleted_count = 0;
    }

    fn should_compact(&self) -> bool {
        let total = self.entries.len();
        total > 0 && (self.deleted_count as f32 / total as f32) > self.max_deleted_ratio
    }

    /// Number of live entries.
    pub fn size(&self) -> usize {
        self.slots.len()
    }

    /// Distances from `q` to every live entry accepted by `filter`.
    fn candidates(&self, q: &[f32], filter: Option<&Filter>) -> PyResult<Vec<(i64, f32)>> {
        let accept = |slot: usize, id: i64| filter.map_or(true, |f| f.accepts(id, slot));
        if let Distance::Custom(name) = &self.metric {
            // Custom metrics may call into Python, so evaluate sequentially.
            let func = get_distance_function_safe(name).map_err(|e| {
                RustAnnError::py_err(
                    "Metric Not Found",
                    format!(
                        "{}. Register it with rust_annie.register_metric() first.",
                        e
                    ),
                )
            })?;
            let mut out = Vec::with_capacity(self.slots.len());
            for (slot, entry) in self.entries.iter().enumerate() {
                if let Some((id, v, _)) = entry {
                    if accept(slot, *id) {
                        let d = func
                            .try_distance(q, v)
                            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
                        out.push((*id, d));
                    }
                }
            }
            return Ok(out);
        }
        let q_norm = l2_norm(q);
        Ok(self
            .entries
            .par_iter()
            .with_min_len(1024)
            .enumerate()
            .filter_map(|(slot, entry)| {
                let (id, v, v_norm) = entry.as_ref()?;
                if !accept(slot, *id) {
                    return None;
                }
                Some((*id, builtin_distance(&self.metric, q, q_norm, v, *v_norm)))
            })
            .collect())
    }

    /// Exact k-NN search for a single query.
    pub fn search_vec(
        &self,
        q: &[f32],
        k: usize,
        filter: Option<&Filter>,
    ) -> PyResult<(Vec<i64>, Vec<f32>)> {
        search::check_k(k)?;
        search::check_query(q, self.dim)?;
        let start = Instant::now();
        let result = top_k(self.candidates(q, filter)?, k);
        if let Some(m) = &self.metrics {
            m.record_query(start.elapsed());
        }
        Ok(result)
    }

    /// Exact k-NN search for a batch of queries. Returns flat row-major
    /// `(n * k)` buffers padded with `-1` / `inf` where fewer than `k`
    /// neighbours exist.
    pub fn search_batch_vec(
        &self,
        queries: ArrayView2<f32>,
        k: usize,
        filter: Option<&Filter>,
    ) -> PyResult<(Vec<i64>, Vec<f32>)> {
        search::check_k(k)?;
        search::check_queries(&queries, self.dim)?;
        let n = queries.nrows();
        let start = Instant::now();
        let per_query: Vec<(Vec<i64>, Vec<f32>)> = if self.metric.is_custom() {
            queries
                .rows()
                .into_iter()
                .map(|row| Ok(top_k(self.candidates(&row.to_vec(), filter)?, k)))
                .collect::<PyResult<_>>()?
        } else {
            // Parallelise across queries; scan entries sequentially per query.
            let accept = |slot: usize, id: i64| filter.map_or(true, |f| f.accepts(id, slot));
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let q = queries.row(i).to_vec();
                    let q_norm = l2_norm(&q);
                    let cands = self
                        .entries
                        .iter()
                        .enumerate()
                        .filter_map(|(slot, entry)| {
                            let (id, v, v_norm) = entry.as_ref()?;
                            if !accept(slot, *id) {
                                return None;
                            }
                            Some((*id, builtin_distance(&self.metric, &q, q_norm, v, *v_norm)))
                        })
                        .collect();
                    top_k(cands, k)
                })
                .collect()
        };
        let mut ids = Vec::with_capacity(n * k);
        let mut dists = Vec::with_capacity(n * k);
        for (i, d) in per_query {
            search::push_padded(&mut ids, &mut dists, i, d, k);
        }
        if let Some(m) = &self.metrics {
            let elapsed = start.elapsed();
            for _ in 0..n {
                m.record_query(elapsed / n.max(1) as u32);
            }
        }
        Ok((ids, dists))
    }

    /// Turn the Python `filter` argument into a [`Filter`].
    ///
    /// Accepts `None`, a `Filter`, or a callable `f(id) -> bool`. Callables are
    /// evaluated once per live entry before the (parallel) search starts.
    pub fn resolve_filter(&self, filter: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Filter>> {
        let Some(obj) = filter else { return Ok(None) };
        if obj.is_none() {
            return Ok(None);
        }
        if let Ok(f) = obj.extract::<Filter>() {
            return Ok(Some(f));
        }
        if obj.is_callable() {
            let mut allowed = Vec::new();
            for entry in self.entries.iter().flatten() {
                if obj.call1((entry.0,))?.is_truthy()? {
                    allowed.push(entry.0);
                }
            }
            return Ok(Some(Filter::id_set(allowed)));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "filter must be None, a rust_annie.Filter, or a callable taking an id and returning bool",
        ))
    }

    /// Serialize to `path`.
    pub fn save_to(&self, path: &Path) -> PyResult<()> {
        let data = BruteForceData {
            dim: self.dim,
            metric: self.metric.clone(),
            max_deleted_ratio: self.max_deleted_ratio,
            entries: self
                .entries
                .iter()
                .flatten()
                .map(|(id, v, _)| (*id, v.clone()))
                .collect(),
        };
        storage::save_brute(path, &data).map_err(|e| e.into_pyerr())
    }

    /// Deserialize from `path`.
    pub fn load_from(path: &Path) -> PyResult<Self> {
        let data = storage::load_brute(path).map_err(|e| e.into_pyerr())?;
        let mut idx = AnnIndex::create(data.dim, data.metric)?;
        idx.max_deleted_ratio = data.max_deleted_ratio;
        for (id, v) in data.entries {
            if v.len() != idx.dim || idx.slots.contains_key(&id) {
                return Err(
                    RustAnnError::io_err(format!("{} is corrupt", path.display())).into_pyerr(),
                );
            }
            let norm = l2_norm(&v);
            idx.slots.insert(id, idx.entries.len());
            idx.entries.push(Some((id, v, norm)));
        }
        Ok(idx)
    }

    pub(crate) fn validate_integrity(&self) -> PyResult<()> {
        let mut errors = Vec::new();
        let mut seen = HashSet::new();
        let mut live = 0;
        for (slot, entry) in self.entries.iter().enumerate() {
            if let Some((id, v, norm)) = entry {
                live += 1;
                if !seen.insert(*id) {
                    errors.push(format!("duplicate ID {}", id));
                }
                if v.len() != self.dim {
                    errors.push(format!(
                        "ID {} has dimension {}, expected {}",
                        id,
                        v.len(),
                        self.dim
                    ));
                }
                if (l2_norm(v) - norm).abs() > 1e-3 * norm.max(1.0) {
                    errors.push(format!("ID {} has a stale cached norm", id));
                }
                if self.slots.get(id) != Some(&slot) {
                    errors.push(format!("ID {} is missing from the ID map", id));
                }
            }
        }
        if live != self.slots.len() {
            errors.push(format!(
                "ID map has {} entries but {} vectors are stored",
                self.slots.len(),
                live
            ));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(RustAnnError::py_err(
                "ValidationError",
                format!("{} issues found: {}", errors.len(), errors.join("; ")),
            ))
        }
    }

    pub(crate) fn info(&self) -> HashMap<String, String> {
        let mut info = HashMap::new();
        info.insert("type".into(), "brute".into());
        info.insert("dim".into(), self.dim.to_string());
        info.insert("metric".into(), self.metric.name());
        info.insert("size".into(), self.size().to_string());
        info.insert("capacity".into(), self.entries.len().to_string());
        info.insert("deleted_count".into(), self.deleted_count.to_string());
        info.insert(
            "max_deleted_ratio".into(),
            self.max_deleted_ratio.to_string(),
        );
        info.insert(
            "version".into(),
            self.version.load(AtomicOrdering::Relaxed).to_string(),
        );
        let memory = self.entries.capacity() * std::mem::size_of::<Option<Entry>>()
            + self.size() * self.dim * std::mem::size_of::<f32>();
        info.insert("memory_bytes".into(), memory.to_string());
        info
    }
}

/// Convert flat padded batch buffers to `(n, k)` NumPy arrays.
pub(crate) fn batch_to_numpy(
    py: Python<'_>,
    n: usize,
    k: usize,
    ids: Vec<i64>,
    dists: Vec<f32>,
) -> PyResult<(PyObject, PyObject)> {
    let ids = Array2::from_shape_vec((n, k), ids)
        .map_err(|e| RustAnnError::py_err("Reshape Error", e.to_string()))?;
    let dists = Array2::from_shape_vec((n, k), dists)
        .map_err(|e| RustAnnError::py_err("Reshape Error", e.to_string()))?;
    Ok((
        ids.to_pyarray(py).into_any().unbind(),
        dists.to_pyarray(py).into_any().unbind(),
    ))
}

pub(crate) fn to_numpy(py: Python<'_>, ids: Vec<i64>, dists: Vec<f32>) -> (PyObject, PyObject) {
    (
        numpy::PyArray1::from_vec(py, ids).into_any().unbind(),
        numpy::PyArray1::from_vec(py, dists).into_any().unbind(),
    )
}

// ---------------------------------------------------------------------------
// Python API
// ---------------------------------------------------------------------------
#[pymethods]
impl AnnIndex {
    #[new]
    #[pyo3(signature = (dim, metric = Distance::Euclidean()))]
    pub fn new(dim: usize, metric: Distance) -> PyResult<Self> {
        Self::create(dim, metric)
    }

    /// Create an index using Minkowski-p distance.
    #[staticmethod]
    pub fn new_minkowski(dim: usize, p: f32) -> PyResult<Self> {
        Self::create(dim, Distance::Minkowski(p))
    }

    /// Create an index using a metric looked up by name. Names that are not
    /// built-in refer to metrics registered with `register_metric`.
    #[staticmethod]
    pub fn new_with_metric(dim: usize, metric_name: &str) -> PyResult<Self> {
        let metric = Distance::new(metric_name);
        if let Distance::Custom(name) = &metric {
            get_distance_function_safe(name).map_err(|_| {
                RustAnnError::py_err(
                    "Metric Not Found",
                    format!(
                        "Unknown metric '{}'. Register it with rust_annie.register_metric() first.",
                        name
                    ),
                )
            })?;
        }
        Self::create(dim, metric)
    }

    /// Add vectors with integer IDs.
    ///
    /// Parameters
    /// ----------
    /// data : array_like, shape (n, dim)
    ///     Vectors to add. Converted to float32.
    /// ids : array_like, shape (n,)
    ///     Unique, non-negative integer IDs.
    pub fn add(
        &mut self,
        data: PyArrayLike2<'_, f32, AllowTypeChange>,
        ids: PyArrayLike1<'_, i64, AllowTypeChange>,
    ) -> PyResult<()> {
        let ids = ids.as_array().to_vec();
        self.add_rows(data.as_array(), &ids)
    }

    /// Add vectors, calling `progress_callback(done, total)` after each chunk.
    #[pyo3(signature = (data, ids, progress_callback, chunk_size = 10_000))]
    pub fn add_batch_with_progress(
        &mut self,
        data: PyArrayLike2<'_, f32, AllowTypeChange>,
        ids: PyArrayLike1<'_, i64, AllowTypeChange>,
        progress_callback: &Bound<'_, PyAny>,
        chunk_size: usize,
    ) -> PyResult<()> {
        let view = data.as_array();
        let ids = ids.as_array().to_vec();
        // Validate everything up front so a failure part-way is impossible.
        search::check_insert(&view, &ids, self.dim, |id| self.slots.contains_key(&id))?;
        let total = ids.len();
        let chunk = chunk_size.max(1);
        let mut done = 0;
        while done < total {
            let end = (done + chunk).min(total);
            self.add_rows(view.slice(ndarray::s![done..end, ..]), &ids[done..end])?;
            done = end;
            progress_callback.call1((done, total))?;
        }
        Ok(())
    }

    /// Find the `k` nearest neighbours of `query`.
    ///
    /// Parameters
    /// ----------
    /// query : array_like, shape (dim,)
    /// k : int
    /// filter : Filter or callable, optional
    ///     Restrict results to IDs accepted by a `Filter` or by a callable
    ///     `f(id) -> bool`.
    ///
    /// Returns
    /// -------
    /// ids : ndarray of int64, shape (m,)
    /// distances : ndarray of float32, shape (m,)
    ///     Sorted by increasing distance; `m = min(k, number of matches)`.
    #[pyo3(signature = (query, k, filter = None))]
    pub fn search(
        &self,
        py: Python<'_>,
        query: PyArrayLike1<'_, f32, AllowTypeChange>,
        k: usize,
        filter: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(PyObject, PyObject)> {
        let q = query.as_array().to_vec();
        let filter = self.resolve_filter(filter)?;
        let (ids, dists) = if self.metric.is_custom() {
            self.search_vec(&q, k, filter.as_ref())?
        } else {
            py.allow_threads(|| self.search_vec(&q, k, filter.as_ref()))?
        };
        Ok(to_numpy(py, ids, dists))
    }

    /// Find the `k` nearest neighbours for each row of `queries`.
    ///
    /// Returns `(ids, distances)` arrays of shape `(n, k)`. Rows with fewer
    /// than `k` matches are padded with id `-1` and distance `inf`.
    #[pyo3(signature = (queries, k, filter = None))]
    pub fn search_batch(
        &self,
        py: Python<'_>,
        queries: PyArrayLike2<'_, f32, AllowTypeChange>,
        k: usize,
        filter: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(PyObject, PyObject)> {
        let view = queries.as_array();
        let n = view.nrows();
        let filter = self.resolve_filter(filter)?;
        let (ids, dists) = if self.metric.is_custom() {
            self.search_batch_vec(view, k, filter.as_ref())?
        } else {
            py.allow_threads(|| self.search_batch_vec(view, k, filter.as_ref()))?
        };
        batch_to_numpy(py, n, k, ids, dists)
    }

    /// Remove vectors by ID. Raises `KeyError` if any ID is missing.
    pub fn remove(&mut self, ids: Vec<i64>) -> PyResult<()> {
        self.remove_ids(&ids)
    }

    /// Replace the vector stored under `id`. Raises `KeyError` if missing.
    pub fn update(
        &mut self,
        id: i64,
        vector: PyArrayLike1<'_, f32, AllowTypeChange>,
    ) -> PyResult<()> {
        let v = vector.as_array().to_vec();
        self.update_vector(id, &v)
    }

    /// Reclaim storage left by removed entries. Called automatically once
    /// the fraction of removed slots exceeds `max_deleted_ratio`.
    pub fn compact(&mut self) {
        self.compact_entries();
        self.bump_version();
    }

    /// Set the fraction of removed slots (0..1) that triggers auto-compaction.
    pub fn set_max_deleted_ratio(&mut self, ratio: f32) -> PyResult<()> {
        if !(0.0..=1.0).contains(&ratio) {
            return Err(RustAnnError::py_err(
                "Invalid Ratio",
                "Ratio must be between 0.0 and 1.0",
            ));
        }
        self.max_deleted_ratio = ratio;
        Ok(())
    }

    /// Whether `id` is stored in the index.
    pub fn __contains__(&self, id: i64) -> bool {
        self.slots.contains_key(&id)
    }

    /// IDs of all stored vectors, in storage order.
    pub fn ids(&self, py: Python<'_>) -> PyObject {
        let ids: Vec<i64> = self.entries.iter().flatten().map(|e| e.0).collect();
        numpy::PyArray1::from_vec(py, ids).into_any().unbind()
    }

    /// Save the index to `path` (any file name; nothing is appended).
    pub fn save(&self, path: &str) -> PyResult<()> {
        let p = validate_path_secure(path)?;
        self.save_to(&p)
    }

    /// Load an index previously written with `save`.
    #[staticmethod]
    pub fn load(path: &str) -> PyResult<Self> {
        let p = validate_path_secure(path)?;
        Self::load_from(&p)
    }

    /// Number of stored vectors.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.size()
    }

    pub fn __len__(&self) -> usize {
        self.size()
    }

    /// Number of storage slots, including removed entries not yet compacted.
    pub fn capacity(&self) -> usize {
        self.entries.len()
    }

    /// Number of removed entries awaiting compaction.
    pub fn deleted_count(&self) -> usize {
        self.deleted_count
    }

    /// Counter incremented on every mutation.
    pub fn version(&self) -> u64 {
        self.version.load(AtomicOrdering::Relaxed)
    }

    /// Vector dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Distance metric used by this index.
    #[getter]
    pub fn metric(&self) -> Distance {
        self.metric.clone()
    }

    pub fn __repr__(&self) -> String {
        format!(
            "AnnIndex(dim={}, metric={}, size={})",
            self.dim,
            self.metric.name(),
            self.size()
        )
    }

    /// Start collecting query metrics; optionally serve them in Prometheus
    /// format at `http://127.0.0.1:<port>/metrics`.
    #[pyo3(signature = (port = None))]
    pub fn enable_metrics(&mut self, port: Option<u16>) -> PyResult<()> {
        let metrics = Arc::new(MetricsCollector::new());
        metrics.set_index_metadata(self.size(), self.dim, &self.metric.name());
        if let Some(p) = port {
            let server = crate::monitoring::MetricsServer::new(Arc::clone(&metrics), p);
            server.start().map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to start metrics server: {}",
                    e
                ))
            })?;
        }
        self.metrics = Some(metrics);
        Ok(())
    }

    /// Snapshot of collected metrics as a dict, or `None` if not enabled.
    pub fn get_metrics(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        let Some(metrics) = &self.metrics else {
            return Ok(None);
        };
        let snap = metrics.get_metrics_snapshot();
        let d = pyo3::types::PyDict::new(py);
        d.set_item("query_count", snap.query_count)?;
        d.set_item("avg_query_latency_us", snap.avg_query_latency_us)?;
        d.set_item("index_size", snap.index_size)?;
        d.set_item("dimensions", snap.dimensions)?;
        d.set_item("uptime_seconds", snap.uptime_seconds)?;
        d.set_item("distance_metric", snap.distance_metric)?;
        d.set_item("recall_estimates", snap.recall_estimates)?;
        Ok(Some(d.into_any().unbind()))
    }

    /// Record an externally measured recall@k value.
    pub fn update_recall_estimate(&self, k: usize, recall: f64) {
        if let Some(metrics) = &self.metrics {
            metrics.update_recall_estimate(k, recall);
        }
    }

    /// Summary of the index configuration and memory use.
    pub fn get_info(&self) -> HashMap<String, String> {
        self.info()
    }

    /// Check internal consistency; raises `ValueError` describing any issues.
    pub fn validate(&self) -> PyResult<()> {
        self.validate_integrity()
    }
}
