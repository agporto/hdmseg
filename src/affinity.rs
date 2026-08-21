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
use crate::vendored::reduce;
use nalgebra::DMatrix;

/// One symmetric weight per undirected graph edge, in `Graph::edge_slice`
/// order. Off-graph entries are exactly zero and are never materialized.
#[derive(Debug, Clone)]
pub struct EdgeWeights {
    values: Vec<f64>,
}

impl EdgeWeights {
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    #[cfg(test)]
    pub fn to_dense(&self, graph: &Graph) -> DMatrix<f64> {
        let mut w = DMatrix::zeros(graph.len(), graph.len());
        for (edge, &(p, q)) in graph.edge_slice().iter().enumerate() {
            let value = self.values[edge];
            w[(p, q)] = value;
            w[(q, p)] = value;
        }
        w
    }
}

/// Per-specimen edge affinities. Stability bootstraps reuse these rows
/// instead of recomputing distances, local medians, and exponentials.
#[derive(Debug, Clone)]
pub struct AffinityCache {
    rows: Vec<Vec<f64>>,
}

impl AffinityCache {
    pub fn build(stack: &Stack, graph: &Graph, parallel: bool) -> Result<Self> {
        validate_graph(stack, graph)?;
        let rows = reduce::map_indexed(stack.n(), parallel, |i| {
            specimen_affinities(stack.specimen(i), graph)
        });
        Ok(Self { rows })
    }

    pub fn consensus(&self, graph: &Graph, subset: &[usize]) -> Result<EdgeWeights> {
        validate_subset(subset, self.rows.len())?;
        let mut accum = vec![0.0_f64; graph.edge_slice().len()];
        // Preserve the legacy bootstrap's exact specimen-index accumulation
        // order, including repeated indices.
        for &i in subset {
            let row = &self.rows[i];
            for edge in 0..accum.len() {
                accum[edge] += row[edge];
            }
        }
        finish_consensus(accum, subset.len())
    }
}

fn validate_graph(stack: &Stack, graph: &Graph) -> Result<()> {
    if graph.len() != stack.m() {
        return Err(HdmError::InvalidConfig(format!(
            "graph has {} loci but stack has {}",
            graph.len(),
            stack.m()
        )));
    }
    Ok(())
}

fn validate_subset(subset: &[usize], n: usize) -> Result<()> {
    if subset.is_empty() {
        return Err(HdmError::InvalidConfig(
            "consensus subset must not be empty".into(),
        ));
    }
    if let Some(&index) = subset.iter().find(|&&index| index >= n) {
        return Err(HdmError::InvalidConfig(format!(
            "consensus subset index {index} is out of range for N={n}"
        )));
    }
    Ok(())
}

fn finish_consensus(mut accum: Vec<f64>, count: usize) -> Result<EdgeWeights> {
    if count == 0 {
        return Err(HdmError::InvalidConfig(
            "consensus subset must not be empty".into(),
        ));
    }
    let inv_n = 1.0 / count as f64;
    for value in &mut accum {
        *value *= inv_n;
    }
    Ok(EdgeWeights { values: accum })
}

/// Build the sparse consensus weights while preserving legacy summation
/// order. Per-specimen affinity rows are independent and may be computed in
/// parallel in bounded batches.
pub fn consensus_edges(
    stack: &Stack,
    graph: &Graph,
    subset: &[usize],
    parallel: bool,
) -> Result<EdgeWeights> {
    validate_graph(stack, graph)?;
    validate_subset(subset, stack.n())?;
    let e = graph.edge_slice().len();
    let mut accum = vec![0.0_f64; e];
    let batch = if parallel {
        rayon::current_num_threads().max(1) * 2
    } else {
        1
    };
    for indices in subset.chunks(batch) {
        let rows = reduce::map_indexed(indices.len(), parallel, |j| {
            specimen_affinities(stack.specimen(indices[j]), graph)
        });
        for row in rows {
            for edge in 0..e {
                accum[edge] += row[edge];
            }
        }
    }
    finish_consensus(accum, subset.len())
}

fn specimen_affinities(x: &DMatrix<f64>, graph: &Graph) -> Vec<f64> {
    let edges = graph.edge_slice();
    let mut dist2 = vec![0.0_f64; edges.len()];
    for (edge, &(p, q)) in edges.iter().enumerate() {
        dist2[edge] = row_dist2(x, p, q);
    }

    let floor = 1e-12_f64;
    let sigma: Vec<f64> = (0..graph.len())
        .map(|p| {
            let incident = graph.incident_edges(p);
            if incident.is_empty() {
                return floor;
            }
            let mut dists: Vec<f64> = incident.iter().map(|&edge| dist2[edge].sqrt()).collect();
            let middle = dists.len() / 2;
            dists.select_nth_unstable_by(middle, |a, b| a.total_cmp(b));
            dists[middle].max(floor)
        })
        .collect();

    let mut values = dist2;
    for (edge, &(p, q)) in edges.iter().enumerate() {
        values[edge] = -values[edge] / (sigma[p] * sigma[q]);
    }
    fastexp::exp_non_positive(&mut values);
    values
}

/// Build the consensus operator `W` (dense, symmetric, `M × M`, nonzero only
/// on graph edges) from the specimens indexed by `subset`. Passing
/// `0..N` gives the full-sample operator; a bootstrap resample passes a
/// multiset of indices.
#[cfg(any(test, feature = "verification"))]
pub fn consensus(stack: &Stack, graph: &Graph, subset: &[usize]) -> Result<DMatrix<f64>> {
    let m = stack.m();
    validate_graph(stack, graph)?;
    validate_subset(subset, stack.n())?;
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
#[cfg(any(test, feature = "verification"))]
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
