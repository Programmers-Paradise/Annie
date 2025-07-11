use pyo3::prelude::*;
use pyo3::PyObject;

#[allow(dead_code)]
pub struct PythonFilter {
    callback: PyObject,
}

#[allow(dead_code)]
impl PythonFilter {
    pub fn new(callback: PyObject) -> Self {
        PythonFilter { callback }
    }

    /// Check if the given ID passes the Python filter
    pub fn accepts(&self, py: Python<'_>, id: i64) -> PyResult<bool> {
        let result = self.callback.call1(py, (id,))?;
        result.extract::<bool>(py)
    }
}
