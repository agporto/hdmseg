//! The consensus (v1) diffusion operator.
//!
//! For each specimen we build a self-tuning Gaussian affinity on the fixed
//! neighbor topology, using that specimen's *own* loci positions, then
//! average across the (sub)sample. Because the bandwidth is per specimen and
//! local, per-specimen scale differences wash out — no normalization step is
//! needed on a pre-aligned-but-unnormalized stack.

use crate::data::Stack;
use crate::error::{HdmError, Result};
use crate::graph::Graph;
use crate::vendored::fastexp;
use nalgebra::DMatrix;

/// Build the consensus operator `W` (dense, symmetric, `M × M`, nonzero only
/// on graph edges) from the specimens indexed by `subset`. Passing
/// `0..N` gives the full-sample operator; a bootstrap resample passes a
/// multiset of indices.
pub fn consensus(stack: &Stack, graph: &Graph, subset: &[usize]) -> Result<DMatrix<f64>> {
    let m = stack.m();
    if graph.len() != m {
        return Err(HdmError::InvalidConfig(format!(
            "graph has {} loci but stack has {m}",
            graph.len()
        )));
    }
    if subset.is_empty() {
        return Err(HdmError::InvalidConfig(
            "consensus subset must not be empty".into(),
        ));
    }
    if let Some(&index) = subset.iter().find(|&&index| index >= stack.n()) {
        return Err(HdmError::InvalidConfig(format!(
            "consensus subset index {index} is out of range for N={}",
            stack.n()
        )));
    }
    let edges: Vec<(usize, usize)> = graph.edges().collect();
    let e = edges.len();
    let mut accum = vec![0.0_f64; e];

    // Global length floor so a degenerate (coincident) neighborhood cannot
    // produce a 0/0 bandwidth.
    let floor = 1e-12_f64;

    let mut arg = vec![0.0_f64; e];
    for &i in subset {
        let x = stack.specimen(i);
        // Project-specific robust local bandwidth: median graph-neighbor
        // distance in this specimen (not the k-th-neighbor rule).
        let sigma = bandwidths(x, graph, floor);
        for (idx, &(p, q)) in edges.iter().enumerate() {
            let dist2 = row_dist2(x, p, q);
            arg[idx] = -dist2 / (sigma[p] * sigma[q]);
        }
        fastexp::exp_non_positive(&mut arg);
        for idx in 0..e {
            accum[idx] += arg[idx];
        }
    }

    let inv_n = 1.0 / subset.len() as f64;
    let mut w = DMatrix::zeros(m, m);
    for (idx, &(p, q)) in edges.iter().enumerate() {
        let val = accum[idx] * inv_n;
        w[(p, q)] = val;
        w[(q, p)] = val;
    }
    Ok(w)
}

/// Median neighbor distance (Euclidean, not squared) per locus, floored.
fn bandwidths(x: &DMatrix<f64>, graph: &Graph, floor: f64) -> Vec<f64> {
    let m = x.nrows();
    (0..m)
        .map(|p| {
            let nbrs = graph.neighbors(p);
            if nbrs.is_empty() {
                return floor;
            }
            let mut dists: Vec<f64> = nbrs.iter().map(|&q| row_dist2(x, p, q).sqrt()).collect();
            dists.sort_by(|a, b| a.total_cmp(b));
            let med = dists[dists.len() / 2];
            med.max(floor)
        })
        .collect()
}

/// Squared Euclidean distance between rows `p` and `q` of `x`.
#[inline]
fn row_dist2(x: &DMatrix<f64>, p: usize, q: usize) -> f64 {
    let d = x.ncols();
    let mut s = 0.0;
    for c in 0..d {
        let diff = x[(p, c)] - x[(q, c)];
        s += diff * diff;
    }
    s
}
