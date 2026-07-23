//! Symmetric normalization of the consensus operator.
//!
//! We form the symmetric normalized operator `S = D^{-1/2} W D^{-1/2}` (real,
//! self-adjoint — what the faer eigensolver wants and deterministic). The
//! random-walk / diffusion-map coordinates are recovered later from `S`'s
//! eigenvectors via the same `D^{-1/2}` factor (see `spectral`).

use nalgebra::{DMatrix, DVector};

/// The normalized operator plus the degree vector needed to map eigenvectors
/// back to diffusion coordinates.
pub struct Normalized {
    /// `S = D^{-1/2} W D^{-1/2}`, dense symmetric `M × M`.
    pub s: DMatrix<f64>,
    /// Degrees `D[p] = sum_q W[p, q]`.
    pub degree: DVector<f64>,
}

/// Normalize `w`. Isolated loci (degree 0) are given a self-degree of 1 so
/// the transform stays finite; they emerge as their own singleton in the
/// embedding.
pub fn normalize(w: &DMatrix<f64>) -> Normalized {
    let m = w.nrows();
    debug_assert_eq!(w.ncols(), m);
    let mut degree = DVector::zeros(m);
    let mut isolated = vec![false; m];
    for p in 0..m {
        degree[p] = w.row(p).sum();
        if degree[p] <= 0.0 {
            degree[p] = 1.0;
            isolated[p] = true;
        }
    }
    let inv_sqrt: Vec<f64> = (0..m).map(|p| 1.0 / degree[p].sqrt()).collect();

    let mut s = DMatrix::zeros(m, m);
    for p in 0..m {
        for q in 0..m {
            let v = w[(p, q)];
            if v != 0.0 {
                s[(p, q)] = inv_sqrt[p] * v * inv_sqrt[q];
            }
        }
        if isolated[p] {
            s[(p, p)] = 1.0;
        }
    }
    Normalized { s, degree }
}
