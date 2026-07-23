//! Python bindings for the `hdmseg` crate.
//!
//! Compiled against the CPython stable ABI (`abi3-py39`), so one wheel per
//! platform serves every CPython ≥ 3.9. The segmentation releases the GIL
//! while the Rust core runs.

use hdmseg::{Config, SelectSpec, Stack};
use nalgebra::DMatrix;
use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use hdmseg as core;

fn err(e: core::HdmError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// The result of a segmentation. Attributes are materialized lazily as NumPy
/// arrays.
#[pyclass]
struct Segmentation {
    labels: Vec<i64>,
    k: usize,
    embedding: DMatrix<f64>,
    eigenvalues: Vec<f64>,
    modularity: f64,
    stability: Option<f64>,
}

#[pymethods]
impl Segmentation {
    /// `(M,)` region id per locus.
    #[getter]
    fn labels(&self, py: Python<'_>) -> Py<PyArray1<i64>> {
        PyArray1::from_slice(py, &self.labels).unbind()
    }

    /// Chosen number of regions.
    #[getter]
    fn k(&self) -> usize {
        self.k
    }

    /// `(M, min(n_components, M-1))` diffusion-map coordinates.
    #[getter]
    fn embedding(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        let (m, c) = (self.embedding.nrows(), self.embedding.ncols());
        Array2::from_shape_fn((m, c), |(i, j)| self.embedding[(i, j)])
            .to_pyarray(py)
            .unbind()
    }

    /// Leading eigenvalues of the normalized operator (descending).
    #[getter]
    fn eigenvalues(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        PyArray1::from_slice(py, &self.eigenvalues).unbind()
    }

    /// Newman-Girvan modularity of the returned partition.
    #[getter]
    fn modularity(&self) -> f64 {
        self.modularity
    }

    /// Bootstrap stability of the returned `k` (`None` unless
    /// `select="stability"`).
    #[getter]
    fn stability(&self) -> Option<f64> {
        self.stability
    }

    fn __repr__(&self) -> String {
        format!(
            "Segmentation(k={}, M={}, modularity={:.4}{})",
            self.k,
            self.labels.len(),
            self.modularity,
            match self.stability {
                Some(s) => format!(", stability={s:.4}"),
                None => String::new(),
            }
        )
    }
}

fn parse_select(
    name: &str,
    k: Option<usize>,
    max_k: usize,
    n_boot: usize,
    seed: u64,
) -> PyResult<SelectSpec> {
    if let Some(k) = k {
        return Ok(SelectSpec::Fixed(k));
    }
    match name {
        "eigengap" => Ok(SelectSpec::Eigengap { max_k }),
        "modularity" => Ok(SelectSpec::Modularity { max_k }),
        "stability" => Ok(SelectSpec::Stability {
            max_k,
            n_boot,
            seed,
        }),
        other => Err(PyValueError::new_err(format!(
            "select must be 'stability', 'modularity', or 'eigengap', got '{other}'"
        ))),
    }
}

/// Segment a correspondence stack. `data` is the flattened `(N, M, D)` array
/// in row-major `(specimen, locus, coord)` order; `reference` is an optional
/// flattened `(M, D)` array (else the plain mean of loci is used).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    data, n, m, d,
    n_neighbors=12, n_components=20, diffusion_time=1.0,
    select="stability", k=None, max_k=12, n_boot=20,
    reference=None, seed=0, parallel=true,
))]
fn segment(
    py: Python<'_>,
    data: PyReadonlyArray1<'_, f64>,
    n: usize,
    m: usize,
    d: usize,
    n_neighbors: usize,
    n_components: usize,
    diffusion_time: f64,
    select: &str,
    k: Option<usize>,
    max_k: usize,
    n_boot: usize,
    reference: Option<PyReadonlyArray1<'_, f64>>,
    seed: u64,
    parallel: bool,
) -> PyResult<Segmentation> {
    let data_slice = data.as_slice()?;
    let reference_mat = match &reference {
        Some(r) => {
            let rs = r.as_slice()?;
            if rs.len() != m * d {
                return Err(PyValueError::new_err(format!(
                    "reference has {} values, expected M*D = {}",
                    rs.len(),
                    m * d
                )));
            }
            Some(DMatrix::from_fn(m, d, |p, c| rs[p * d + c]))
        }
        None => None,
    };

    let stack = Stack::from_flat(data_slice, n, m, d, reference_mat).map_err(err)?;

    let cfg = Config {
        n_neighbors,
        n_components,
        diffusion_time,
        select: parse_select(select, k, max_k, n_boot, seed)?,
        seed,
        parallel,
    };

    let seg = py.detach(|| core::segment(&stack, &cfg)).map_err(err)?;

    Ok(Segmentation {
        labels: seg.labels.iter().map(|&x| x as i64).collect(),
        k: seg.k,
        embedding: seg.embedding,
        eigenvalues: seg.eigenvalues,
        modularity: seg.modularity,
        stability: seg.stability,
    })
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Segmentation>()?;
    m.add_function(wrap_pyfunction!(segment, m)?)?;
    Ok(())
}
