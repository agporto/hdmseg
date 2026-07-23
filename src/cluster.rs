//! Deterministic Ng-Jordan-Weiss spectral clustering.
//!
//! We row-normalize the leading `k` eigenvectors of the symmetric normalized
//! operator and run Euclidean k-means on those features. Initialization is
//! seeded k-means++ with a splitmix64 stream, and every tie is broken by the
//! lowest index, so labels are deterministic in `(vectors, k, seed)`.

use nalgebra::DMatrix;

#[inline]
fn split_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn split_unit(state: &mut u64) -> f64 {
    (split_next(state) >> 11) as f64 / (1u64 << 53) as f64
}

#[inline]
fn dist2(x: &DMatrix<f64>, p: usize, center: &[f64]) -> f64 {
    let d = x.ncols();
    let mut s = 0.0;
    for c in 0..d {
        let diff = x[(p, c)] - center[c];
        s += diff * diff;
    }
    s
}

/// Row-normalize the first `n_dims` columns of `vectors` to unit L2 (NJW).
fn njw_features(vectors: &DMatrix<f64>, n_dims: usize) -> DMatrix<f64> {
    let m = vectors.nrows();
    let d = n_dims.min(vectors.ncols()).max(1);
    DMatrix::from_fn(m, d, |p, c| {
        let norm: f64 = (0..d)
            .map(|k| vectors[(p, k)] * vectors[(p, k)])
            .sum::<f64>()
            .sqrt();
        if norm > 1e-300 {
            vectors[(p, c)] / norm
        } else {
            0.0
        }
    })
}

/// Cluster into `k` groups using the Ng-Jordan-Weiss recipe: take the top
/// `k` raw eigenvectors of `S` (columns of `vectors`), row-normalize, and
/// run deterministic k-means. Returns a label in `0..k` per locus.
/// Deterministic in `(vectors, k, seed)`.
pub fn kmeans(vectors: &DMatrix<f64>, k: usize, seed: u64) -> Vec<usize> {
    let feats = njw_features(vectors, k);
    let x = &feats;
    let (m, d) = (x.nrows(), x.ncols());
    let k = k.min(m).max(1);
    let mut state = seed ^ 0xD1B5_4A32_D192_ED03;

    // ---- k-means++ seeding ----
    let mut centers: Vec<Vec<f64>> = Vec::with_capacity(k);
    let first = (split_next(&mut state) % m as u64) as usize;
    centers.push((0..d).map(|c| x[(first, c)]).collect());

    let mut best_d2 = vec![0.0_f64; m];
    for p in 0..m {
        best_d2[p] = dist2(x, p, &centers[0]);
    }
    while centers.len() < k {
        let total: f64 = best_d2.iter().sum();
        let next = if total <= 0.0 {
            // All remaining points coincide with a center: pick the lowest
            // index not yet a center (deterministic).
            (split_next(&mut state) % m as u64) as usize
        } else {
            let target = split_unit(&mut state) * total;
            let mut acc = 0.0;
            let mut chosen = m - 1;
            for p in 0..m {
                acc += best_d2[p];
                if acc >= target {
                    chosen = p;
                    break;
                }
            }
            chosen
        };
        let new_center: Vec<f64> = (0..d).map(|c| x[(next, c)]).collect();
        for p in 0..m {
            let dd = dist2(x, p, &new_center);
            if dd < best_d2[p] {
                best_d2[p] = dd;
            }
        }
        centers.push(new_center);
    }

    // ---- Lloyd iterations ----
    let mut labels = vec![0usize; m];
    for _iter in 0..100 {
        let mut changed = false;
        for p in 0..m {
            let mut best = 0usize;
            let mut best_v = f64::INFINITY;
            for (ci, center) in centers.iter().enumerate() {
                let dd = dist2(x, p, center);
                if dd < best_v {
                    best_v = dd;
                    best = ci;
                }
            }
            if labels[p] != best {
                changed = true;
            }
            labels[p] = best;
        }

        // Recompute centers; handle empties by seizing the farthest point.
        let mut sums = vec![vec![0.0_f64; d]; k];
        let mut counts = vec![0usize; k];
        for p in 0..m {
            let c = labels[p];
            counts[c] += 1;
            for j in 0..d {
                sums[c][j] += x[(p, j)];
            }
        }
        for ci in 0..k {
            if counts[ci] > 0 {
                for j in 0..d {
                    centers[ci][j] = sums[ci][j] / counts[ci] as f64;
                }
            } else {
                // Farthest point from its own center becomes the new center.
                let mut far = 0usize;
                let mut far_v = -1.0;
                for p in 0..m {
                    let dd = dist2(x, p, &centers[labels[p]]);
                    if dd > far_v {
                        far_v = dd;
                        far = p;
                    }
                }
                centers[ci] = (0..d).map(|j| x[(far, j)]).collect();
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    // Guarantee the public Fixed(k) contract even for degenerate features
    // where several centers coincide. Move the deterministic farthest point
    // from a non-singleton cluster into each empty cluster.
    let mut counts = vec![0usize; k];
    for &label in &labels {
        counts[label] += 1;
    }
    for empty in 0..k {
        if counts[empty] != 0 {
            continue;
        }
        let mut donor_point = None;
        let mut donor_dist = -1.0;
        for p in 0..m {
            let donor = labels[p];
            if counts[donor] <= 1 {
                continue;
            }
            let dd = dist2(x, p, &centers[donor]);
            if dd > donor_dist {
                donor_dist = dd;
                donor_point = Some(p);
            }
        }
        let p = donor_point.expect("k <= M guarantees a donor for every empty cluster");
        counts[labels[p]] -= 1;
        labels[p] = empty;
        counts[empty] = 1;
    }

    labels
}
