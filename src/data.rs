//! The correspondence stack: `N` specimens sharing `M` loci in
//! correspondence, in `D` dimensions.

use crate::error::{HdmError, Result};
use nalgebra::DMatrix;

/// A population of homologous shapes already in dense correspondence.
///
/// `specimens[i]` is an `M × D` matrix of that specimen's loci positions;
/// row `p` of every specimen refers to the same corresponding locus. The
/// `reference` is an `M × D` representative shape used only for the fixed
/// neighbor topology (plain mean of loci by default — the builder is assumed
/// to have pre-aligned the specimens, so no GPA).
#[derive(Debug, Clone)]
pub struct Stack {
    n: usize,
    m: usize,
    d: usize,
    specimens: Vec<DMatrix<f64>>,
    reference: DMatrix<f64>,
}

impl Stack {
    /// Build from a flat, row-major `(N, M, D)` buffer (specimen-major, then
    /// locus, then coordinate). If `reference` is `None`, the plain mean of
    /// the loci across specimens is used.
    pub fn from_flat(
        data: &[f64],
        n: usize,
        m: usize,
        d: usize,
        reference: Option<DMatrix<f64>>,
    ) -> Result<Self> {
        if n == 0 || m < 2 {
            return Err(HdmError::InvalidStack(format!(
                "need N >= 1 and M >= 2, got N={n}, M={m}"
            )));
        }
        if d != 2 && d != 3 {
            return Err(HdmError::InvalidStack(format!("D must be 2 or 3, got {d}")));
        }
        let expected = n
            .checked_mul(m)
            .and_then(|nm| nm.checked_mul(d))
            .ok_or_else(|| HdmError::InvalidStack("N*M*D overflows usize".into()))?;
        if data.len() != expected {
            return Err(HdmError::InvalidStack(format!(
                "coordinate buffer has {} values, expected N*M*D = {}",
                data.len(),
                expected
            )));
        }
        if let Some(index) = data.iter().position(|value| !value.is_finite()) {
            return Err(HdmError::InvalidStack(format!(
                "coordinate buffer contains a non-finite value at flat index {index}"
            )));
        }
        let specimens: Vec<DMatrix<f64>> = (0..n)
            .map(|i| DMatrix::from_fn(m, d, |p, c| data[(i * m + p) * d + c]))
            .collect();

        let reference = match reference {
            Some(r) => {
                if r.nrows() != m || r.ncols() != d {
                    return Err(HdmError::InvalidStack(format!(
                        "reference is {}x{}, expected {m}x{d}",
                        r.nrows(),
                        r.ncols()
                    )));
                }
                if let Some(index) = r.iter().position(|value| !value.is_finite()) {
                    return Err(HdmError::InvalidStack(format!(
                        "reference contains a non-finite value at column-major index {index}"
                    )));
                }
                r
            }
            None => {
                let mut mean = DMatrix::zeros(m, d);
                for s in &specimens {
                    mean += s;
                }
                mean /= n as f64;
                mean
            }
        };

        Ok(Self {
            n,
            m,
            d,
            specimens,
            reference,
        })
    }

    /// Number of specimens.
    pub fn n(&self) -> usize {
        self.n
    }
    /// Number of loci.
    pub fn m(&self) -> usize {
        self.m
    }
    /// Dimensionality (2 or 3).
    pub fn d(&self) -> usize {
        self.d
    }
    /// The `M × D` positions of specimen `i`.
    pub(crate) fn specimen(&self, i: usize) -> &DMatrix<f64> {
        &self.specimens[i]
    }
    /// The `M × D` reference shape (graph topology substrate).
    pub fn reference(&self) -> &DMatrix<f64> {
        &self.reference
    }
    /// All specimen matrices.
    pub fn specimens(&self) -> &[DMatrix<f64>] {
        &self.specimens
    }
}
