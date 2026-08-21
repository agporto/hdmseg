//! Deterministic k-nearest-neighbor queries on a small point set.
//!
//! rustcpd uses `kiddo` for its sparse E-step; here the reference loci count
//! `M` is in the hundreds to low thousands, so a brute-force scan is simpler,
//! fully deterministic, and dimension-agnostic. If `M` ever grows large
//! enough to matter, replace this implementation with a deterministic spatial
//! index and validate that neighbor tie-breaking remains stable.

use nalgebra::DMatrix;

/// For each row of `points` (an `M × D` matrix), return the indices of its
/// `k` nearest neighbors (excluding itself), sorted by increasing squared
/// distance with the point index as a deterministic tie-break.
pub(crate) fn knn(points: &DMatrix<f64>, k: usize, parallel: bool) -> Vec<Vec<usize>> {
    let m = points.nrows();
    let d = points.ncols();
    let k = k.min(m.saturating_sub(1));
    super::reduce::map_indexed(m, parallel, |i| {
        let mut scored: Vec<(f64, usize)> = Vec::with_capacity(m - 1);
        for j in 0..m {
            if j == i {
                continue;
            }
            let mut dist2 = 0.0;
            for c in 0..d {
                let diff = points[(i, c)] - points[(j, c)];
                dist2 += diff * diff;
            }
            scored.push((dist2, j));
        }
        let cmp = |a: &(f64, usize), b: &(f64, usize)| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1));
        if k < scored.len() {
            scored.select_nth_unstable_by(k, cmp);
            scored.truncate(k);
        }
        scored.sort_by(cmp);
        scored.truncate(k);
        scored.into_iter().map(|(_, j)| j).collect()
    })
}
