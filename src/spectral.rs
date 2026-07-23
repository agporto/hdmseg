//! Spectral embedding: leading eigenvectors of the symmetric normalized
//! operator.
//!
//! Two products come out of the same eigendecomposition:
//!
//! * `vectors` — the top `n_components` **raw** eigenvectors of `S`
//!   (descending, column 0 is the trivial one). These drive clustering via
//!   the Ng-Jordan-Weiss recipe (top-`k`, row-normalized), which stays
//!   correct even when the graph is disconnected (each connected component
//!   contributes an eigenvalue-1 indicator, so dropping the trivial vector
//!   would throw away a genuine separator).
//! * `coords` — the requested number of random-walk **diffusion-map**
//!   coordinates (`|λ|^t · D^{-1/2} · u`, trivial component dropped), returned
//!   to the caller.
//!
use crate::diffusion::Normalized;
use crate::error::{HdmError, Result};
use crate::vendored::linalg;
use nalgebra::DMatrix;

/// The spectral products.
pub struct Embedding {
    /// `M × n_components` raw eigenvectors of `S` (descending; col 0 trivial).
    pub vectors: DMatrix<f64>,
    /// Leading eigenvalues of `S`, descending (`len n_components`).
    pub eigenvalues: Vec<f64>,
    /// `M × requested_dimensions` diffusion-map coordinates.
    pub coords: DMatrix<f64>,
}

fn inv_sqrt_degree(norm: &Normalized) -> Vec<f64> {
    (0..norm.degree.len())
        .map(|p| {
            let d = norm.degree[p];
            if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 }
        })
        .collect()
}

/// Assemble `Embedding` from raw eigenvectors/eigenvalues already ordered
/// descending and truncated to `n_components` columns.
fn assemble(
    vectors: DMatrix<f64>,
    eigenvalues: Vec<f64>,
    inv_sqrt: &[f64],
    coord_dims: usize,
    diffusion_time: f64,
) -> Embedding {
    let m = vectors.nrows();
    let nc = vectors.ncols();
    let coords = DMatrix::from_fn(m, coord_dims, |p, j| {
        let col = j + 1; // drop trivial column 0
        debug_assert!(col < nc);
        let scale = eigenvalues[col].abs().powf(diffusion_time);
        scale * inv_sqrt[p] * vectors[(p, col)]
    });
    Embedding {
        vectors,
        eigenvalues,
        coords,
    }
}

/// Dense path: full symmetric eigendecomposition of `S`, keep the leading
/// `n_components` eigenvectors (including the trivial one).
pub fn dense(
    norm: &Normalized,
    n_vectors: usize,
    coord_dims: usize,
    diffusion_time: f64,
    parallel: bool,
) -> Result<Embedding> {
    let m = norm.s.nrows();
    let nc = n_vectors.min(m);
    debug_assert!(coord_dims < nc);
    let (all_vectors, values) =
        linalg::symmetric_eigen(&norm.s, parallel).ok_or(HdmError::EigenFailed)?;

    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| values[b].total_cmp(&values[a]));
    order.truncate(nc);

    let vectors = DMatrix::from_fn(m, nc, |p, j| all_vectors[(p, order[j])]);
    let eigenvalues: Vec<f64> = order.iter().map(|&c| values[c]).collect();

    let inv_sqrt = inv_sqrt_degree(norm);
    Ok(assemble(
        vectors,
        eigenvalues,
        &inv_sqrt,
        coord_dims,
        diffusion_time,
    ))
}
