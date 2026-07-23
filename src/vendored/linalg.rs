//! Thin bridge to `faer` for dense GEMM and the symmetric eigensolve.
//!
//! VENDORED from `rustcpd/src/solve.rs` (`matmul_into`, `symmetric_eigen`,
//! `par`). Do not edit for hdmseg-specific reasons; sync from rustcpd.

use faer::Accum;
use faer::Par;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::{self, ComputeEigenvectors};
use faer::linalg::matmul::matmul;
use faer::mat::{MatMut, MatRef};
use nalgebra::DMatrix;

pub(crate) fn par(parallel: bool) -> Par {
    if parallel { Par::rayon(0) } else { Par::Seq }
}

/// Write `dst = opᵃ(a) · opᵇ(b)` using faer's matmul, where `opᵃ`/`opᵇ`
/// optionally transpose their operand. `dst` must be preallocated with the
/// result dimensions; no allocation happens here.
#[allow(dead_code)]
pub(crate) fn matmul_into(
    dst: &mut DMatrix<f64>,
    a: &DMatrix<f64>,
    a_transposed: bool,
    b: &DMatrix<f64>,
    b_transposed: bool,
    parallel: bool,
) {
    let (m, n) = (dst.nrows(), dst.ncols());
    let a_ref = MatRef::from_column_major_slice(a.as_slice(), a.nrows(), a.ncols());
    let b_ref = MatRef::from_column_major_slice(b.as_slice(), b.nrows(), b.ncols());
    let a_op = if a_transposed {
        a_ref.transpose()
    } else {
        a_ref
    };
    let b_op = if b_transposed {
        b_ref.transpose()
    } else {
        b_ref
    };
    let dst_mut = MatMut::from_column_major_slice_mut(dst.as_mut_slice(), m, n);
    matmul(dst_mut, Accum::Replace, a_op, b_op, 1.0, par(parallel));
}

/// Full symmetric eigendecomposition of `g` (assumed self-adjoint) via
/// faer. Returns `(eigenvectors, eigenvalues)` with eigenvalues ascending
/// (faer's order); columns of the matrix are the corresponding vectors.
pub(crate) fn symmetric_eigen(
    g: &DMatrix<f64>,
    parallel: bool,
) -> Option<(DMatrix<f64>, Vec<f64>)> {
    let m = g.nrows();
    let par = par(parallel);
    let mut s = faer::diag::Diag::<f64>::zeros(m);
    let mut u = faer::Mat::<f64>::zeros(m, m);
    let mut buffer = MemBuffer::new(evd::self_adjoint_evd_scratch::<f64>(
        m,
        ComputeEigenvectors::Yes,
        par,
        Default::default(),
    ));
    evd::self_adjoint_evd(
        MatRef::from_column_major_slice(g.as_slice(), m, m),
        s.as_mut(),
        Some(u.as_mut()),
        par,
        MemStack::new(&mut buffer),
        Default::default(),
    )
    .ok()?;
    let vectors = DMatrix::from_fn(m, m, |i, j| u[(i, j)]);
    let values = (0..m).map(|i| s[i]).collect();
    Some((vectors, values))
}
