//! Feature-gated hooks for the independent numerical verification script.
//!
//! This module is not part of the supported application API. It is available
//! only with the `verification` Cargo feature.

use crate::affinity;
use crate::diffusion;
use crate::graph::Graph;
use crate::spectral;
use crate::{HdmError, Result, Stack};
use nalgebra::DMatrix;

/// Consensus operator and leading dense eigenvalues used by `verify/check.py`.
pub struct VerificationOutput {
    pub operator: DMatrix<f64>,
    pub eigenvalues: Vec<f64>,
}

/// Build the consensus operator and dense spectrum for independent checking.
pub fn operator_and_spectrum(
    stack: &Stack,
    n_neighbors: usize,
    n_eigenvalues: usize,
) -> Result<VerificationOutput> {
    if n_neighbors == 0 {
        return Err(HdmError::InvalidConfig("n_neighbors must be >= 1".into()));
    }
    if n_eigenvalues == 0 {
        return Err(HdmError::InvalidConfig("n_eigenvalues must be >= 1".into()));
    }

    let graph = Graph::knn(stack, n_neighbors, false);
    let subset: Vec<usize> = (0..stack.n()).collect();
    let operator = affinity::consensus(stack, &graph, &subset)?;
    let normalized = diffusion::normalize(&operator);
    let embedding = spectral::dense(&normalized, n_eigenvalues.min(stack.m()), 0, 1.0, false)?;

    Ok(VerificationOutput {
        operator,
        eigenvalues: embedding.eigenvalues,
    })
}
