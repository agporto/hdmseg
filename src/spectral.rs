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
use crate::diffusion::{Normalized, NormalizedEdges};
use crate::error::{HdmError, Result};
use crate::graph::Graph;
use crate::vendored::linalg;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::matrix_free::eigen::{
    PartialEigenParams, partial_self_adjoint_eigen, partial_self_adjoint_eigen_scratch,
};
use faer::sparse::{SparseRowMat, Triplet};
use faer::{Col, Mat};
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

fn inv_sqrt_degree_edges(norm: &NormalizedEdges) -> Vec<f64> {
    norm.degree()
        .iter()
        .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 })
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

/// Minimum locus count at which avoiding the dense `M × M` decomposition is
/// worthwhile. The dense path is also the exact compatibility fallback for
/// small and difficult spectra.
pub const SPARSE_MIN_LOCI: usize = 256;

/// Whether faer's single-vector Krylov-Schur solver is applicable without
/// changing the requested invariant subspace. A disconnected graph can have
/// repeated eigenvalue 1; a single-vector Krylov method cannot recover the
/// full multiplicity reliably, so those cases retain the dense path.
pub fn sparse_applicable(norm: &NormalizedEdges, graph: &Graph, n_vectors: usize) -> bool {
    let m = graph.len();
    if m < SPARSE_MIN_LOCI || n_vectors >= m || m <= 64.max(2 * n_vectors) {
        return false;
    }
    let mut seen = vec![false; m];
    let mut stack = vec![0usize];
    seen[0] = true;
    while let Some(p) = stack.pop() {
        for &edge in graph.incident_edges(p) {
            if norm.values()[edge] == 0.0 {
                continue;
            }
            let (a, b) = graph.edge_slice()[edge];
            let q = if a == p { b } else { a };
            if !seen[q] {
                seen[q] = true;
                stack.push(q);
            }
        }
    }
    seen.into_iter().all(|value| value)
}

/// Partial eigendecomposition of the sparse normalized operator.
///
/// faer's partial solver targets eigenvalues with largest magnitude. We solve
/// `(I + S)u = μu` instead: normalized-affinity eigenvalues lie in `[-1, 1]`,
/// so the largest-magnitude `μ` are exactly the largest-algebraic eigenvalues
/// `λ = μ - 1` requested by the dense implementation.
pub fn sparse(
    norm: &NormalizedEdges,
    graph: &Graph,
    n_vectors: usize,
    coord_dims: usize,
    diffusion_time: f64,
    parallel: bool,
) -> Result<Embedding> {
    let m = graph.len();
    let nc = n_vectors.min(m);
    debug_assert!(coord_dims < nc);
    if !sparse_applicable(norm, graph, nc) {
        return Err(HdmError::EigenFailed);
    }

    let mut triplets = Vec::with_capacity(m + 2 * graph.edge_slice().len());
    for p in 0..m {
        // S has a unit self-loop only for isolated nodes.
        let diagonal = if norm.isolated()[p] { 2.0 } else { 1.0 };
        triplets.push(Triplet::new(p, p, diagonal));
    }
    for (edge, &(p, q)) in graph.edge_slice().iter().enumerate() {
        let value = norm.values()[edge];
        if value != 0.0 {
            triplets.push(Triplet::new(p, q, value));
            triplets.push(Triplet::new(q, p, value));
        }
    }
    let operator = SparseRowMat::<usize, f64>::try_new_from_triplets(m, m, &triplets)
        .map_err(|_| HdmError::EigenFailed)?;

    // A wider subspace is important for surfaces, whose leading modes often
    // arrive in tight harmonic clusters. It uses O(M · max_dim) memory but
    // avoids hundreds of narrow restart cycles.
    let params = PartialEigenParams {
        min_dim: (2 * nc).max(48),
        max_dim: (8 * nc).max(256).min(m - 1),
        max_restarts: 200,
        ..PartialEigenParams::default()
    };

    let par = linalg::par(parallel);
    let mut v0 = Col::<f64>::zeros(m);
    let mut state = 0xD6E8_FEB8_6659_FD93_u64 ^ m as u64 ^ ((nc as u64) << 32);
    for p in 0..m {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let bits = (z ^ (z >> 31)) >> 11;
        v0[p] = bits as f64 / (1_u64 << 53) as f64 - 0.5;
    }

    let mut raw_vectors = Mat::<f64>::zeros(m, nc);
    let mut shifted_values = vec![0.0_f64; nc];
    let mut buffer = MemBuffer::new(partial_self_adjoint_eigen_scratch(
        &operator, nc, par, params,
    ));
    let workspace = MemStack::new(&mut buffer);
    let info = partial_self_adjoint_eigen(
        raw_vectors.as_mut(),
        &mut shifted_values,
        &operator,
        v0.as_ref(),
        1e-12,
        par,
        workspace,
        params,
    );
    if info.n_converged_eigen != nc {
        return Err(HdmError::EigenFailed);
    }

    let mut order: Vec<usize> = (0..nc).collect();
    order.sort_by(|&a, &b| shifted_values[b].total_cmp(&shifted_values[a]));
    let eigenvalues: Vec<f64> = order
        .iter()
        .map(|&column| shifted_values[column] - 1.0)
        .collect();
    let vectors = DMatrix::from_fn(m, nc, |p, j| raw_vectors[(p, order[j])]);

    // Independently check the returned Ritz pairs before exposing them. Auto
    // mode falls back to the dense reference if this guard ever trips.
    for j in 0..nc {
        let lambda = eigenvalues[j];
        let mut residual2 = 0.0;
        for p in 0..m {
            let mut sv = if norm.isolated()[p] {
                vectors[(p, j)]
            } else {
                0.0
            };
            for &edge in graph.incident_edges(p) {
                let (a, b) = graph.edge_slice()[edge];
                let q = if a == p { b } else { a };
                sv += norm.values()[edge] * vectors[(q, j)];
            }
            let residual = sv - lambda * vectors[(p, j)];
            residual2 += residual * residual;
        }
        if !residual2.is_finite() || residual2.sqrt() > 1e-8 * lambda.abs().max(1.0) {
            return Err(HdmError::EigenFailed);
        }
    }
    for j in 0..nc {
        for k in 0..=j {
            let mut dot = 0.0;
            for p in 0..m {
                dot += vectors[(p, j)] * vectors[(p, k)];
            }
            let expected = if j == k { 1.0 } else { 0.0 };
            if !dot.is_finite() || (dot - expected).abs() > 1e-8 {
                return Err(HdmError::EigenFailed);
            }
        }
    }

    let inv_sqrt = inv_sqrt_degree_edges(norm);
    Ok(assemble(
        vectors,
        eigenvalues,
        &inv_sqrt,
        coord_dims,
        diffusion_time,
    ))
}
