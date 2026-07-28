//! Symmetric normalization of the consensus operator.
//!
//! We form the symmetric normalized operator `S = D^{-1/2} W D^{-1/2}` (real,
//! self-adjoint — what the faer eigensolver wants and deterministic). The
//! random-walk / diffusion-map coordinates are recovered later from `S`'s
//! eigenvectors via the same `D^{-1/2}` factor (see `spectral`).

use crate::affinity::EdgeWeights;
use crate::graph::Graph;
use nalgebra::{DMatrix, DVector};

/// The normalized operator plus the degree vector needed to map eigenvectors
/// back to diffusion coordinates.
pub struct Normalized {
    /// `S = D^{-1/2} W D^{-1/2}`, dense symmetric `M × M`.
    pub s: DMatrix<f64>,
    /// Degrees `D[p] = sum_q W[p, q]`.
    pub degree: DVector<f64>,
}

/// Sparse normalized operator: one value per undirected graph edge plus
/// explicit isolated-node self-loops.
pub struct NormalizedEdges {
    values: Vec<f64>,
    degree: Vec<f64>,
    isolated: Vec<bool>,
}

impl NormalizedEdges {
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    pub fn degree(&self) -> &[f64] {
        &self.degree
    }

    pub fn isolated(&self) -> &[bool] {
        &self.isolated
    }

    pub fn to_dense(&self, graph: &Graph) -> Normalized {
        let m = graph.len();
        let mut s = DMatrix::zeros(m, m);
        for (edge, &(p, q)) in graph.edge_slice().iter().enumerate() {
            let value = self.values[edge];
            s[(p, q)] = value;
            s[(q, p)] = value;
        }
        for p in 0..m {
            if self.isolated[p] {
                s[(p, p)] = 1.0;
            }
        }
        Normalized {
            s,
            degree: DVector::from_column_slice(&self.degree),
        }
    }
}

/// Normalize edge weights without materializing structural zeros. Degree
/// sums follow the same per-row neighbor order as the dense implementation.
pub fn normalize_edges(w: &EdgeWeights, graph: &Graph) -> NormalizedEdges {
    let m = graph.len();
    debug_assert_eq!(w.values().len(), graph.edge_slice().len());
    let mut degree = vec![0.0_f64; m];
    let mut isolated = vec![false; m];
    for p in 0..m {
        for &edge in graph.incident_edges(p) {
            degree[p] += w.values()[edge];
        }
        if degree[p] <= 0.0 {
            degree[p] = 1.0;
            isolated[p] = true;
        }
    }
    let inv_sqrt: Vec<f64> = degree.iter().map(|&value| 1.0 / value.sqrt()).collect();
    let mut values = vec![0.0; graph.edge_slice().len()];
    for (edge, &(p, q)) in graph.edge_slice().iter().enumerate() {
        let value = w.values()[edge];
        // The legacy dense eigensolver reads only the lower triangle.  Since
        // p < q for every cached edge, retain that exact floating-point
        // multiplication order here as well.
        values[edge] = inv_sqrt[q] * value * inv_sqrt[p];
    }
    NormalizedEdges {
        values,
        degree,
        isolated,
    }
}

/// Normalize `w`. Isolated loci (degree 0) are given a self-degree of 1 so
/// the transform stays finite; they emerge as their own singleton in the
/// embedding.
#[cfg(any(test, feature = "verification"))]
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
