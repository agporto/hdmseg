//! The fixed neighbor topology: a symmetric kNN graph on the reference loci.

use crate::data::Stack;
use crate::vendored::spatial;

/// Undirected adjacency: `neighbors[p]` lists the loci sharing an edge with
/// locus `p` (sorted, deduplicated, symmetric).
#[derive(Debug, Clone)]
pub struct Graph {
    neighbors: Vec<Vec<usize>>,
}

impl Graph {
    /// Build the symmetric k-nearest-neighbor graph on `stack.reference()`.
    /// An edge `(p, q)` exists if `q` is among `p`'s `k` nearest neighbors
    /// or vice versa (the union symmetrization standard in spectral
    /// clustering).
    pub fn knn(stack: &Stack, k: usize) -> Self {
        let directed = spatial::knn(stack.reference(), k);
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
        Self { neighbors: sets }
    }

    /// Neighbor list of locus `p`.
    pub fn neighbors(&self, p: usize) -> &[usize] {
        &self.neighbors[p]
    }

    /// Number of loci.
    pub fn len(&self) -> usize {
        self.neighbors.len()
    }

    /// Iterate every undirected edge once, as `(p, q)` with `p < q`.
    pub fn edges(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.neighbors.iter().enumerate().flat_map(|(p, nbrs)| {
            nbrs.iter()
                .copied()
                .filter(move |&q| q > p)
                .map(move |q| (p, q))
        })
    }
}
