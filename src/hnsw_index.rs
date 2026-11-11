//! Approximate nearest-neighbour search with HNSW graphs.
//!
//! Graph construction and traversal are delegated to the `hnsw_rs` crate
//! (Malkov & Yashunin, 2018). This module adds user-supplied integer IDs,
//! soft deletion, filtered search, multiple metrics, persistence and the
//! Python bindings.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use hnsw_rs::prelude::*;
use ndarray::ArrayView2;
use numpy::{AllowTypeChange, PyArrayLike1, PyArrayLike2};
use pyo3::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::errors::RustAnnError;
use crate::filters::Filter;
use crate::index::{batch_to_numpy, to_numpy, MAX_DIM};
use crate::metrics::Distance;
use crate::path_validation::validate_path_secure;
use crate::search;
use crate::storage::{self, HnswData};

/// Maximum number of layers in the graph (the `hnsw_rs` limit).
const MAX_LAYERS: usize = 16;

/// HNSW construction and search parameters.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct HnswConfig {
    /// Maximum number of graph neighbours per node (`M`), 2..=256.
    pub m: usize,
    /// Candidate list size during construction. Higher is slower but more accurate.
    pub ef_construction: usize,
    /// Default candidate list size during search (raised to `k` if smaller).
    pub ef_search: usize,
    /// Expected number of elements; a sizing hint, not a hard limit.
    pub max_elements: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
            max_elements: 10_000,
        }
    }
}

impl HnswConfig {
    pub fn validate(&self) -> Result<(), RustAnnError> {
        if !(2..=256).contains(&self.m) {
            return Err(RustAnnError::Message(
                "`m` must be between 2 and 256".into(),
            ));
        }
        if self.ef_construction == 0 {
            return Err(RustAnnError::Message(
                "`ef_construction` must be greater than 0".into(),
            ));
        }
        if self.ef_search == 0 {
            return Err(RustAnnError::Message(
                "`ef_search` must be greater than 0".into(),
            ));
        }
        if self.max_elements == 0 {
            return Err(RustAnnError::Message(
                "`max_elements` must be greater than 0".into(),
            ));
        }
        Ok(())
    }
}

/// The underlying graph, specialised per supported metric.
enum Graph {
    L2(Hnsw<'static, f32, DistL2>),
    Cosine(Hnsw<'static, f32, DistCosine>),
    L1(Hnsw<'static, f32, DistL1>),
}

macro_rules! with_graph {
    ($graph:expr, $g:ident => $body:expr) => {
        match $graph {
            Graph::L2($g) => $body,
            Graph::Cosine($g) => $body,
            Graph::L1($g) => $body,
        }
    };
}

impl Graph {
    fn new(metric: &Distance, config: &HnswConfig) -> Result<Self, RustAnnError> {
        let (m, n, l, ef) = (
            config.m,
            config.max_elements,
            MAX_LAYERS,
            config.ef_construction,
        );
        Ok(match metric {
            Distance::Euclidean() => Graph::L2(Hnsw::new(m, n, l, ef, DistL2 {})),
            Distance::Cosine() => Graph::Cosine(Hnsw::new(m, n, l, ef, DistCosine {})),
            Distance::Manhattan() => Graph::L1(Hnsw::new(m, n, l, ef, DistL1 {})),
            other => {
                return Err(RustAnnError::Message(format!(
                    "HNSW supports EUCLIDEAN, COSINE and MANHATTAN metrics, not {}",
                    other.name()
                )))
            }
        })
    }
}

/// Approximate k-NN index backed by an HNSW graph.
pub struct HnswIndex {
    graph: Graph,
    dims: usize,
    metric: Distance,
    pub config: HnswConfig,
    /// Internal (graph) ID -> user ID.
    user_ids: Vec<i64>,
    /// Internal ID -> vector, kept for persistence.
    vectors: Vec<Vec<f32>>,
    /// User ID -> internal ID, live entries only.
    lookup: HashMap<i64, usize>,
    /// Internal IDs that have been removed.
    deleted: HashSet<usize>,
}

impl HnswIndex {
    pub fn new(dims: usize, metric: Distance, config: HnswConfig) -> Result<Self, RustAnnError> {
        if dims == 0 || dims > MAX_DIM {
            return Err(RustAnnError::Message(format!(
                "dims must be between 1 and {}",
                MAX_DIM
            )));
        }
        config.validate()?;
        Ok(HnswIndex {
            graph: Graph::new(&metric, &config)?,
            dims,
            metric,
            config,
            user_ids: Vec::new(),
            vectors: Vec::new(),
            lookup: HashMap::new(),
            deleted: HashSet::new(),
        })
    }

    pub fn dims(&self) -> usize {
        self.dims
    }

    pub fn metric(&self) -> &Distance {
        &self.metric
    }

    /// Number of live (non-removed) vectors.
    pub fn size(&self) -> usize {
        self.lookup.len()
    }

    /// Insert rows of `data` with the given IDs. Validates all input first.
    pub fn add_rows(&mut self, data: ArrayView2<f32>, ids: &[i64]) -> PyResult<()> {
        if matches!(self.metric, Distance::Cosine())
            && data.rows().into_iter().any(|r| r.iter().all(|x| *x == 0.0))
        {
            // `hnsw_rs` treats a zero vector as identical to every other vector,
            // which would put it at the top of every result list.
            return Err(RustAnnError::py_err(
                "Invalid Input",
                "cosine distance is undefined for zero vectors; PyHnswIndex with \
                 Distance.COSINE does not accept them (AnnIndex does)",
            ));
        }
        self.insert_rows(data, ids)
    }

    /// Insert rows without the zero-vector check (used when loading files
    /// written before that check existed).
    fn insert_rows(&mut self, data: ArrayView2<f32>, ids: &[i64]) -> PyResult<()> {
        search::check_insert(&data, ids, self.dims, |id| self.lookup.contains_key(&id))?;
        let start = self.user_ids.len();
        for (row, &id) in data.rows().into_iter().zip(ids) {
            self.lookup.insert(id, self.user_ids.len());
            self.user_ids.push(id);
            self.vectors.push(row.to_vec());
        }
        let batch: Vec<(&[f32], usize)> = self.vectors[start..]
            .iter()
            .enumerate()
            .map(|(i, v)| (v.as_slice(), start + i))
            .collect();
        with_graph!(&self.graph, g => {
            if batch.len() > 1 {
                g.parallel_insert_slice(&batch);
            } else {
                for item in &batch {
                    g.insert_slice(*item);
                }
            }
        });
        Ok(())
    }

    /// Soft-delete entries by ID. Raises `KeyError` if any ID is missing.
    pub fn remove_ids(&mut self, ids: &[i64]) -> PyResult<()> {
        let missing: Vec<i64> = ids
            .iter()
            .copied()
            .filter(|id| !self.lookup.contains_key(id))
            .collect();
        if !missing.is_empty() {
            return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "IDs not found in index: {:?}",
                missing
            )));
        }
        for id in ids {
            if let Some(internal) = self.lookup.remove(id) {
                self.deleted.insert(internal);
            }
        }
        Ok(())
    }

    /// Approximate k-NN search for one query.
    pub fn search_vec(
        &self,
        q: &[f32],
        k: usize,
        ef: Option<usize>,
        filter: Option<&Filter>,
    ) -> PyResult<(Vec<i64>, Vec<f32>)> {
        search::check_k(k)?;
        search::check_query(q, self.dims)?;
        if self.size() == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        // Neither k nor ef can usefully exceed the number of stored vectors, and
        // `hnsw_rs` allocates buffers of size ef, so clamp both.
        let k = k.min(self.size());
        let ef = ef
            .unwrap_or(self.config.ef_search)
            .min(self.user_ids.len())
            .max(k);
        let accept = |internal: &usize| -> bool {
            if self.deleted.contains(internal) {
                return false;
            }
            filter.is_none_or(|f| f.accepts(self.user_ids[*internal], *internal))
        };

        // Selective filters starve graph traversal (most neighbours are rejected),
        // which lowers recall. Count the matching vectors first: if there are few
        // of them, an exact scan over just those is both exact and cheap.
        let allowed: Option<Vec<usize>> =
            filter.map(|_| (0..self.user_ids.len()).filter(&accept).collect());

        // Scan exactly when every stored vector fits in the candidate list (graph
        // traversal costs as much and can miss nodes), and for a zero cosine query,
        // whose distance is 1 to every vector (see `AnnIndex`) so the graph cannot rank it.
        let zero_cosine_query =
            matches!(self.metric, Distance::Cosine()) && q.iter().all(|x| *x == 0.0);
        if ef >= self.user_ids.len() || zero_cosine_query {
            let live: Vec<usize> = match allowed {
                Some(a) => a,
                None => (0..self.user_ids.len()).filter(&accept).collect(),
            };
            return Ok(self.exact_over(q, k, &live));
        }
        if let Some(allowed) = &allowed {
            let threshold = (ef * self.config.m * 2).max(self.size() / 10);
            if allowed.len() <= threshold {
                return Ok(self.exact_over(q, k, allowed));
            }
        }

        let needs_filter = filter.is_some() || !self.deleted.is_empty();
        let neighbours = with_graph!(&self.graph, g => {
            if needs_filter {
                g.search_filter(q, k, ef, Some(&accept))
            } else {
                g.search(q, k, ef)
            }
        });
        let candidates: Vec<(i64, f32)> = neighbours
            .into_iter()
            .filter(|n| accept(&n.d_id))
            .map(|n| (self.user_ids[n.d_id], n.distance))
            .collect();
        // Safety net: if traversal found fewer than k matches although more exist,
        // fall back to an exact scan over the matching vectors.
        if let Some(allowed) = &allowed {
            if candidates.len() < k.min(allowed.len()) {
                return Ok(self.exact_over(q, k, allowed));
            }
        }
        Ok(search::top_k(candidates, k))
    }

    /// Exact k-NN over the given internal IDs, using the index metric.
    fn exact_over(&self, q: &[f32], k: usize, internal_ids: &[usize]) -> (Vec<i64>, Vec<f32>) {
        let q_norm = search::l2_norm(q);
        let candidates = internal_ids
            .iter()
            .map(|&i| {
                let v = &self.vectors[i];
                let d = search::builtin_distance(&self.metric, q, q_norm, v, search::l2_norm(v));
                (self.user_ids[i], d)
            })
            .collect();
        search::top_k(candidates, k)
    }

    /// Approximate k-NN search for a batch of queries; flat padded output.
    pub fn search_batch_vec(
        &self,
        queries: ArrayView2<f32>,
        k: usize,
        ef: Option<usize>,
        filter: Option<&Filter>,
    ) -> PyResult<(Vec<i64>, Vec<f32>)> {
        search::check_k(k)?;
        search::check_queries(&queries, self.dims)?;
        // Allocate the output first: fail cleanly before searching if it is too large.
        let (mut ids, mut dists) = search::batch_buffers(queries.nrows(), k)?;
        let results: Vec<(Vec<i64>, Vec<f32>)> = (0..queries.nrows())
            .into_par_iter()
            .map(|i| self.search_vec(&queries.row(i).to_vec(), k, ef, filter))
            .collect::<PyResult<_>>()?;
        for (i, d) in results {
            search::push_padded(&mut ids, &mut dists, i, d, k);
        }
        Ok((ids, dists))
    }

    fn live_entries(&self) -> impl Iterator<Item = (i64, &Vec<f32>)> {
        self.user_ids
            .iter()
            .zip(&self.vectors)
            .enumerate()
            .filter(|(internal, _)| !self.deleted.contains(internal))
            .map(|(_, (id, v))| (*id, v))
    }

    /// Resolve the Python `filter` argument (None, `Filter`, or callable).
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
            for (id, _) in self.live_entries() {
                if obj.call1((id,))?.is_truthy()? {
                    allowed.push(id);
                }
            }
            return Ok(Some(Filter::id_set(allowed)));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "filter must be None, a rust_annie.Filter, or a callable taking an id and returning bool",
        ))
    }

    /// Serialize live entries and configuration to `path`.
    pub fn save_to(&self, path: &Path) -> PyResult<()> {
        let data = HnswData {
            dim: self.dims,
            metric: self.metric.clone(),
            config: self.config.clone(),
            entries: self.live_entries().map(|(id, v)| (id, v.clone())).collect(),
        };
        storage::save_hnsw(path, &data).map_err(|e| e.into_pyerr())
    }

    /// Load from `path`, rebuilding the graph from the stored vectors.
    pub fn load_from(path: &Path) -> PyResult<Self> {
        let data = storage::load_hnsw(path).map_err(|e| e.into_pyerr())?;
        let mut config = data.config;
        config.max_elements = config.max_elements.max(data.entries.len());
        let mut idx = HnswIndex::new(data.dim, data.metric, config).map_err(|e| e.into_pyerr())?;
        let ids: Vec<i64> = data.entries.iter().map(|(id, _)| *id).collect();
        let flat: Vec<f32> = data.entries.into_iter().flat_map(|(_, v)| v).collect();
        let n = ids.len();
        let arr = ndarray::Array2::from_shape_vec((n, idx.dims), flat).map_err(|_| {
            RustAnnError::io_err(format!("{} is corrupt", path.display())).into_pyerr()
        })?;
        // Files written by 0.3.0 may contain zero vectors in a cosine index;
        // keep loading them rather than refusing the whole file.
        idx.insert_rows(arr.view(), &ids)?;
        Ok(idx)
    }

    pub fn get_info(&self) -> HashMap<String, String> {
        let mut info = HashMap::new();
        info.insert("type".into(), "hnsw".into());
        info.insert("dim".into(), self.dims.to_string());
        info.insert("metric".into(), self.metric.name());
        info.insert("size".into(), self.size().to_string());
        info.insert("deleted_count".into(), self.deleted.len().to_string());
        info.insert("max_elements".into(), self.config.max_elements.to_string());
        info.insert("m".into(), self.config.m.to_string());
        info.insert(
            "ef_construction".into(),
            self.config.ef_construction.to_string(),
        );
        info.insert("ef_search".into(), self.config.ef_search.to_string());
        let memory = self.vectors.len() * (self.dims * 4 + std::mem::size_of::<Vec<f32>>() + 8);
        info.insert("memory_bytes".into(), memory.to_string());
        info
    }

    pub fn validate(&self) -> Result<(), RustAnnError> {
        let mut errors = Vec::new();
        if self.user_ids.len() != self.vectors.len() {
            errors.push(format!(
                "ID/vector count mismatch: {} IDs vs {} vectors",
                self.user_ids.len(),
                self.vectors.len()
            ));
        }
        if let Some((i, v)) = self
            .vectors
            .iter()
            .enumerate()
            .find(|(_, v)| v.len() != self.dims)
        {
            errors.push(format!(
                "vector {} has dimension {}, expected {}",
                i,
                v.len(),
                self.dims
            ));
        }
        let graph_points = with_graph!(&self.graph, g => g.get_nb_point());
        if graph_points != self.vectors.len() {
            errors.push(format!(
                "graph has {} points but {} vectors are stored",
                graph_points,
                self.vectors.len()
            ));
        }
        if self.lookup.len() + self.deleted.len() != self.user_ids.len() {
            errors.push("live + deleted counts do not match stored entries".into());
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(RustAnnError::Message(format!(
                "{} issues found: {}",
                errors.len(),
                errors.join("; ")
            )))
        }
    }
}

// ---------------------------------------------------------------------------
// Python bindings
// ---------------------------------------------------------------------------

/// HNSW construction and search parameters.
#[pyclass(name = "HnswConfig", module = "rust_annie")]
#[derive(Clone)]
pub struct PyHnswConfig {
    #[pyo3(get, set)]
    pub m: usize,
    #[pyo3(get, set)]
    pub ef_construction: usize,
    #[pyo3(get, set)]
    pub ef_search: usize,
    #[pyo3(get, set)]
    pub max_elements: usize,
}

#[pymethods]
impl PyHnswConfig {
    #[new]
    #[pyo3(signature = (m = 16, ef_construction = 200, ef_search = 50, max_elements = 10_000))]
    pub fn new(m: usize, ef_construction: usize, ef_search: usize, max_elements: usize) -> Self {
        PyHnswConfig {
            m,
            ef_construction,
            ef_search,
            max_elements,
        }
    }

    #[staticmethod]
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Self {
        HnswConfig::default().into()
    }

    pub fn validate(&self) -> PyResult<()> {
        self.to_config()
            .validate()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "HnswConfig(m={}, ef_construction={}, ef_search={}, max_elements={})",
            self.m, self.ef_construction, self.ef_search, self.max_elements
        )
    }
}

impl PyHnswConfig {
    pub fn to_config(&self) -> HnswConfig {
        HnswConfig {
            m: self.m,
            ef_construction: self.ef_construction,
            ef_search: self.ef_search,
            max_elements: self.max_elements,
        }
    }
}

impl From<HnswConfig> for PyHnswConfig {
    fn from(c: HnswConfig) -> Self {
        PyHnswConfig {
            m: c.m,
            ef_construction: c.ef_construction,
            ef_search: c.ef_search,
            max_elements: c.max_elements,
        }
    }
}

/// Approximate k-nearest-neighbour index based on HNSW graphs.
///
/// Parameters
/// ----------
/// dims : int
///     Dimensionality of the vectors.
/// metric : Distance, default Distance.EUCLIDEAN
///     One of EUCLIDEAN, COSINE or MANHATTAN.
/// m : int, default 16
///     Maximum neighbours per graph node.
/// ef_construction : int, default 200
///     Construction-time candidate list size.
/// ef_search : int, default 50
///     Default search-time candidate list size; override per query with `ef`.
/// max_elements : int, default 10000
///     Expected number of elements (sizing hint only).
/// config : HnswConfig, optional
///     Alternative to passing the four parameters individually.
#[pyclass(name = "PyHnswIndex", module = "rust_annie")]
pub struct PyHnswIndex {
    pub(crate) inner: HnswIndex,
}

#[pymethods]
impl PyHnswIndex {
    #[new]
    #[pyo3(signature = (
        dims, metric = Distance::Euclidean(), m = 16, ef_construction = 200,
        ef_search = 50, max_elements = 10_000, config = None
    ))]
    fn new(
        dims: usize,
        metric: Distance,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        max_elements: usize,
        config: Option<PyHnswConfig>,
    ) -> PyResult<Self> {
        let config = config.map(|c| c.to_config()).unwrap_or(HnswConfig {
            m,
            ef_construction,
            ef_search,
            max_elements,
        });
        let inner = HnswIndex::new(dims, metric, config)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyHnswIndex { inner })
    }

    /// Add vectors with unique, non-negative integer IDs.
    fn add(
        &mut self,
        py: Python<'_>,
        data: PyArrayLike2<'_, f32, AllowTypeChange>,
        ids: PyArrayLike1<'_, i64, AllowTypeChange>,
    ) -> PyResult<()> {
        let ids = ids.as_array().to_vec();
        let view = data.as_array();
        let inner = &mut self.inner;
        py.allow_threads(|| inner.add_rows(view, &ids))
    }

    /// Approximate k-NN search for one query.
    ///
    /// `ef` overrides the index's `ef_search` for this call (higher is more
    /// accurate and slower). Returns `(ids, distances)` sorted by distance.
    #[pyo3(signature = (query, k, ef = None, filter = None))]
    fn search(
        &self,
        py: Python<'_>,
        query: PyArrayLike1<'_, f32, AllowTypeChange>,
        k: usize,
        ef: Option<usize>,
        filter: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(PyObject, PyObject)> {
        let q = query.as_array().to_vec();
        let filter = self.inner.resolve_filter(filter)?;
        let inner = &self.inner;
        let (ids, dists) = py.allow_threads(|| inner.search_vec(&q, k, ef, filter.as_ref()))?;
        Ok(to_numpy(py, ids, dists))
    }

    /// Approximate k-NN search for each row of `queries`; returns `(n, k)`
    /// arrays padded with id `-1` / distance `inf`.
    #[pyo3(signature = (queries, k, ef = None, filter = None))]
    fn search_batch(
        &self,
        py: Python<'_>,
        queries: PyArrayLike2<'_, f32, AllowTypeChange>,
        k: usize,
        ef: Option<usize>,
        filter: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(PyObject, PyObject)> {
        let view = queries.as_array();
        let n = view.nrows();
        let filter = self.inner.resolve_filter(filter)?;
        let inner = &self.inner;
        let (ids, dists) =
            py.allow_threads(|| inner.search_batch_vec(view, k, ef, filter.as_ref()))?;
        batch_to_numpy(py, n, k, ids, dists)
    }

    /// Remove vectors by ID (soft delete). Raises `KeyError` if any is missing.
    fn remove(&mut self, ids: Vec<i64>) -> PyResult<()> {
        self.inner.remove_ids(&ids)
    }

    /// Save to `path`. The graph is rebuilt from the vectors on `load`.
    fn save(&self, path: &str) -> PyResult<()> {
        let p = validate_path_secure(path)?;
        self.inner.save_to(&p)
    }

    /// Load an index written by `save`.
    #[staticmethod]
    fn load(py: Python<'_>, path: &str) -> PyResult<Self> {
        let p = validate_path_secure(path)?;
        let inner = py.allow_threads(|| HnswIndex::load_from(&p))?;
        Ok(PyHnswIndex { inner })
    }

    /// Default search-time candidate list size.
    #[getter]
    fn get_ef_search(&self) -> usize {
        self.inner.config.ef_search
    }

    #[setter]
    fn set_ef_search(&mut self, ef: usize) -> PyResult<()> {
        if ef == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "ef_search must be > 0",
            ));
        }
        self.inner.config.ef_search = ef;
        Ok(())
    }

    #[getter]
    fn metric(&self) -> Distance {
        self.inner.metric.clone()
    }

    #[getter]
    fn config(&self) -> PyHnswConfig {
        self.inner.config.clone().into()
    }

    fn dims(&self) -> usize {
        self.inner.dims
    }

    fn len(&self) -> usize {
        self.inner.size()
    }

    fn __len__(&self) -> usize {
        self.inner.size()
    }

    fn __contains__(&self, id: i64) -> bool {
        self.inner.lookup.contains_key(&id)
    }

    fn get_info(&self) -> HashMap<String, String> {
        self.inner.get_info()
    }

    fn validate(&self) -> PyResult<()> {
        self.inner
            .validate()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "PyHnswIndex(dims={}, metric={}, size={}, m={}, ef_construction={}, ef_search={})",
            self.inner.dims,
            self.inner.metric.name(),
            self.inner.size(),
            self.inner.config.m,
            self.inner.config.ef_construction,
            self.inner.config.ef_search
        )
    }
}
