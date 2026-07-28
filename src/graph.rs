//! The fixed neighbor topology: a symmetric kNN graph on the reference loci.

use crate::data::Stack;
use crate::vendored::spatial;

/// Undirected adjacency: `neighbors[p]` lists the loci sharing an edge with
/// locus `p` (sorted, deduplicated, symmetric).
#[derive(Debug, Clone)]
pub struct Graph {
    neighbors: Vec<Vec<usize>>,
    edges: Vec<(usize, usize)>,
    incident_edges: Vec<Vec<usize>>,
}

impl Graph {
    /// Build the symmetric k-nearest-neighbor graph on `stack.reference()`.
    /// An edge `(p, q)` exists if `q` is among `p`'s `k` nearest neighbors
    /// or vice versa (the union symmetrization standard in spectral
    /// clustering).
    pub fn knn(stack: &Stack, k: usize, parallel: bool) -> Self {
        let directed = spatial::knn(stack.reference(), k, parallel);
        let m = directed.len();
        let mut sets: Vec<Vec<usize>> = vec![Vec::new(); m];
        for (p, nbrs) in directed.iter().enumerate() {
            for &q in nbrs {
                sets[p].push(q);
                sets[q].push(p);
            }
        }
        for s in &mut sets {
            s.sort_unstable();
            s.dedup();
        }
        let mut edges = Vec::new();
        let mut incident_edges = vec![Vec::new(); m];
        for (p, nbrs) in sets.iter().enumerate() {
            for &q in nbrs {
                if q > p {
                    let edge = edges.len();
                    edges.push((p, q));
                    incident_edges[p].push(edge);
                    incident_edges[q].push(edge);
                }
            }
        }
        Self {
            neighbors: sets,
            edges,
            incident_edges,
        }
    }

    /// Neighbor list of locus `p`.
    #[cfg(any(test, feature = "verification"))]
    pub fn neighbors(&self, p: usize) -> &[usize] {
        &self.neighbors[p]
    }

    /// Number of loci.
    pub fn len(&self) -> usize {
        self.neighbors.len()
    }

    /// Undirected edges, sorted lexicographically as `(p, q)` with `p < q`.
    pub fn edge_slice(&self) -> &[(usize, usize)] {
        &self.edges
    }

    /// Edge indices incident on `p`, ordered by the neighboring locus index.
    pub fn incident_edges(&self, p: usize) -> &[usize] {
        &self.incident_edges[p]
    }

    /// Iterate every undirected edge once, as `(p, q)` with `p < q`.
    #[cfg(any(test, feature = "verification"))]
    pub fn edges(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.edges.iter().copied()
    }
}
