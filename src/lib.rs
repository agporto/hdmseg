//! # hdmseg
//!
//! Population-consistent surface segmentation by **correspondence-collapsed
//! consensus diffusion maps**. Given a stack of `N` homologous shapes sharing
//! `M` loci in dense correspondence, hdmseg builds one `M × M` frame-free
//! consensus operator on the loci and applies dense spectral clustering.
//!
//! This is not a horizontal or hypoelliptic diffusion-map implementation.
//!
//! ```no_run
//! use hdmseg::{Stack, Config, segment};
//! # fn demo(flat: &[f64], n: usize, m: usize) {
//! let stack = Stack::from_flat(flat, n, m, 3, None).unwrap();
//! let seg = segment(&stack, &Config::default()).unwrap();
//! let _labels = seg.labels; // region id per locus
//! # }
//! ```

// Numerical kernels index parallel arrays/matrices by position; enumerate()
// would obscure the math without removing a bounds check that matters here.
#![allow(clippy::needless_range_loop)]

mod affinity;
mod cluster;
mod data;
mod diffusion;
mod error;
mod graph;
mod select;
mod spectral;
mod vendored;
#[cfg(feature = "verification")]
#[doc(hidden)]
pub mod verification;

pub use data::Stack;
pub use error::{HdmError, Result};

use graph::Graph;
use nalgebra::DMatrix;
use spectral::Embedding;

/// How to choose the number of regions `k`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectSpec {
    /// Use exactly this many regions.
    Fixed(usize),
    /// Eigengap heuristic over `2..=max_k`.
    Eigengap { max_k: usize },
    /// Maximize Newman-Girvan modularity over `2..=max_k`.
    Modularity { max_k: usize },
    /// Bootstrap stability over `2..=max_k` (the default).
    Stability {
        max_k: usize,
        n_boot: usize,
        seed: u64,
    },
}

/// Full pipeline configuration.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    /// k for the kNN reference graph.
    pub n_neighbors: usize,
    /// Number of non-trivial diffusion-coordinate dimensions to return.
    pub n_components: usize,
    /// Diffusion time `t` (eigenvalue power in the coordinates).
    pub diffusion_time: f64,
    /// Model selection strategy.
    pub select: SelectSpec,
    /// Seed for k-means.
    pub seed: u64,
    /// Use the rayon pool (results are identical to serial).
    pub parallel: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            n_neighbors: 12,
            n_components: 20,
            diffusion_time: 1.0,
            select: SelectSpec::Stability {
                max_k: 12,
                n_boot: 20,
                seed: 0,
            },
            seed: 0,
            parallel: true,
        }
    }
}

/// The result of a segmentation.
#[derive(Debug, Clone)]
pub struct Segmentation {
    /// Region id in `0..k` per locus (`length M`).
    pub labels: Vec<usize>,
    /// The chosen number of regions.
    pub k: usize,
    /// `M × min(n_components, M - 1)` diffusion coordinates.
    pub embedding: DMatrix<f64>,
    /// Leading eigenvalues of the normalized operator (descending).
    pub eigenvalues: Vec<f64>,
    /// Newman-Girvan modularity of the returned partition.
    pub modularity: f64,
    /// Bootstrap stability of the returned `k` (only for `Stability`).
    pub stability: Option<f64>,
}

#[inline]
fn split_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Build the kNN reference graph.
fn build_graph(stack: &Stack, cfg: &Config) -> Graph {
    Graph::knn(stack, cfg.n_neighbors.max(1))
}

fn validate_config(stack: &Stack, cfg: &Config) -> Result<()> {
    let m = stack.m();
    if cfg.n_neighbors == 0 {
        return Err(HdmError::InvalidConfig("n_neighbors must be >= 1".into()));
    }
    if cfg.n_components == 0 {
        return Err(HdmError::InvalidConfig("n_components must be >= 1".into()));
    }
    if !cfg.diffusion_time.is_finite() || cfg.diffusion_time < 0.0 {
        return Err(HdmError::InvalidConfig(
            "diffusion_time must be finite and >= 0".into(),
        ));
    }
    match cfg.select {
        SelectSpec::Fixed(k) => {
            if k == 0 {
                return Err(HdmError::InvalidConfig("fixed k must be >= 1".into()));
            }
            if k > m {
                return Err(HdmError::TooManyClusters {
                    requested: k,
                    available: m,
                });
            }
        }
        SelectSpec::Eigengap { max_k } => {
            if max_k < 2 {
                return Err(HdmError::InvalidConfig(
                    "eigengap max_k must be >= 2".into(),
                ));
            }
            if m < 3 {
                return Err(HdmError::TooManyClusters {
                    requested: 2,
                    available: m.saturating_sub(1),
                });
            }
        }
        SelectSpec::Modularity { max_k } => {
            if max_k < 2 {
                return Err(HdmError::InvalidConfig(
                    "modularity max_k must be >= 2".into(),
                ));
            }
        }
        SelectSpec::Stability { max_k, n_boot, .. } => {
            if max_k < 2 {
                return Err(HdmError::InvalidConfig(
                    "stability max_k must be >= 2".into(),
                ));
            }
            if n_boot == 0 {
                return Err(HdmError::InvalidConfig(
                    "stability n_boot must be >= 1".into(),
                ));
            }
        }
    }
    Ok(())
}

/// Build the consensus operator and its spectral embedding for a specimen
/// subset (`0..N` for the full sample; a resample for bootstrapping).
fn operator_and_embedding(
    stack: &Stack,
    graph: &Graph,
    subset: &[usize],
    n_vectors: usize,
    coord_dims: usize,
    cfg: &Config,
) -> Result<(DMatrix<f64>, Embedding)> {
    let w = affinity::consensus(stack, graph, subset)?;
    let norm = diffusion::normalize(&w);
    let emb = spectral::dense(
        &norm,
        n_vectors,
        coord_dims,
        cfg.diffusion_time,
        cfg.parallel,
    )?;
    Ok((w, emb))
}

/// Segment the stack.
pub fn segment(stack: &Stack, cfg: &Config) -> Result<Segmentation> {
    validate_config(stack, cfg)?;
    let m = stack.m();

    let coord_dims = cfg.n_components.min(m - 1);
    let selection_vectors = match cfg.select {
        SelectSpec::Fixed(k) => k.min(m),
        SelectSpec::Eigengap { max_k } => max_k.min(m - 1) + 1,
        SelectSpec::Modularity { max_k } | SelectSpec::Stability { max_k, .. } => max_k.min(m),
    };
    let n_vectors = (coord_dims + 1).max(selection_vectors).min(m);

    let graph = build_graph(stack, cfg);
    let (w_full, emb_full) = operator_and_embedding(
        stack,
        &graph,
        &full_subset(stack.n()),
        n_vectors,
        coord_dims,
        cfg,
    )?;

    let (k, labels, stability) = match cfg.select {
        SelectSpec::Fixed(k) => (k, cluster::kmeans(&emb_full.vectors, k, cfg.seed), None),
        SelectSpec::Eigengap { max_k } => {
            let hi = max_k.min(m - 1);
            let k = select::eigengap(&emb_full.eigenvalues, hi);
            (k, cluster::kmeans(&emb_full.vectors, k, cfg.seed), None)
        }
        SelectSpec::Modularity { max_k } => {
            let hi = max_k.min(m);
            let mut best_k = 2;
            let mut best_q = f64::NEG_INFINITY;
            let mut best_labels = Vec::new();
            for k in 2..=hi {
                let labels = cluster::kmeans(&emb_full.vectors, k, cfg.seed);
                let q = select::modularity(&w_full, &labels);
                if q > best_q {
                    best_q = q;
                    best_k = k;
                    best_labels = labels;
                }
            }
            (best_k, best_labels, None)
        }
        SelectSpec::Stability {
            max_k,
            n_boot,
            seed,
        } => {
            let hi = max_k.min(m);
            let full_labels: Vec<Vec<usize>> = (2..=hi)
                .map(|k| cluster::kmeans(&emb_full.vectors, k, cfg.seed))
                .collect();

            let n = stack.n();
            let boot = n_boot;

            // Draw every resample's index-set sequentially first, so the RNG
            // stream (and thus the resamples) is fixed regardless of how the
            // work is later scheduled.
            let mut rng = seed ^ 0xA5A5_5A5A_1234_ABCD;
            let subsets: Vec<Vec<usize>> = (0..boot)
                .map(|_| {
                    (0..n)
                        .map(|_| (split_next(&mut rng) % n as u64) as usize)
                        .collect()
                })
                .collect();

            // Process bootstrap operators one at a time to keep dense M × M
            // memory bounded. Individual eigensolves may still use the rayon
            // pool when cfg.parallel is true.
            let score_one = |subset: &Vec<usize>| -> Result<Vec<f64>> {
                let (wb, emb_b) =
                    operator_and_embedding(stack, &graph, subset, n_vectors, coord_dims, cfg)?;
                drop(wb);
                Ok((2..=hi)
                    .enumerate()
                    .map(|(ki, k)| {
                        let labels_b = cluster::kmeans(&emb_b.vectors, k, cfg.seed);
                        select::adjusted_rand_index(&full_labels[ki], &labels_b)
                    })
                    .collect())
            };
            let per_boot: Result<Vec<Vec<f64>>> = subsets.iter().map(&score_one).collect();
            let per_boot = per_boot?;

            // Fixed-order reduction: sum over resamples in index order.
            let mut ari_sum = vec![0.0_f64; hi + 1]; // index by k
            for per in &per_boot {
                for (ki, k) in (2..=hi).enumerate() {
                    ari_sum[k] += per[ki];
                }
            }

            let mut best_k = 2;
            let mut best_score = f64::NEG_INFINITY;
            for k in 2..=hi {
                let score = ari_sum[k] / boot as f64;
                if score > best_score {
                    best_score = score;
                    best_k = k;
                }
            }
            let labels = full_labels[best_k - 2].clone();
            (best_k, labels, Some(best_score))
        }
    };

    let modularity = select::modularity(&w_full, &labels);

    Ok(Segmentation {
        labels,
        k,
        embedding: emb_full.coords,
        eigenvalues: emb_full.eigenvalues,
        modularity,
        stability,
    })
}

fn full_subset(n: usize) -> Vec<usize> {
    (0..n).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two well-separated 3-D blobs of loci, jittered per specimen. A correct
    /// segmentation must recover the two blobs.
    fn two_blob_stack(n: usize, per: usize, jitter: f64) -> Stack {
        let m = 2 * per;
        let d = 3;
        let mut rng = 0xDEAD_BEEF_u64;
        let nx = |s: &mut u64| -> f64 {
            *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = *s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        // Base template: blob A near origin, blob B near (10,0,0).
        let mut base = vec![0.0_f64; m * d];
        for p in 0..per {
            base[p * d] = nx(&mut rng);
            base[p * d + 1] = nx(&mut rng);
            base[p * d + 2] = nx(&mut rng);
        }
        for p in per..m {
            base[p * d] = 10.0 + nx(&mut rng);
            base[p * d + 1] = nx(&mut rng);
            base[p * d + 2] = nx(&mut rng);
        }
        let mut flat = vec![0.0_f64; n * m * d];
        for i in 0..n {
            for idx in 0..m * d {
                flat[i * m * d + idx] = base[idx] + jitter * nx(&mut rng);
            }
        }
        Stack::from_flat(&flat, n, m, d, None).unwrap()
    }

    fn two_regions_correct(labels: &[usize], per: usize) -> bool {
        // All of blob A shares a label; all of blob B shares the other.
        let a = labels[0];
        let b = labels[per];
        a != b && labels[..per].iter().all(|&l| l == a) && labels[per..].iter().all(|&l| l == b)
    }

    #[test]
    fn recovers_two_blobs_fixed_k() {
        let per = 30;
        let stack = two_blob_stack(6, per, 0.05);
        let cfg = Config {
            n_neighbors: 8,
            n_components: 6,
            select: SelectSpec::Fixed(2),
            ..Config::default()
        };
        let seg = segment(&stack, &cfg).unwrap();
        assert_eq!(seg.k, 2);
        assert!(
            two_regions_correct(&seg.labels, per),
            "labels: {:?}",
            seg.labels
        );
    }

    #[test]
    fn stability_selects_two() {
        let per = 30;
        let stack = two_blob_stack(12, per, 0.05);
        let cfg = Config {
            n_neighbors: 8,
            n_components: 6,
            select: SelectSpec::Stability {
                max_k: 6,
                n_boot: 10,
                seed: 0,
            },
            ..Config::default()
        };
        let seg = segment(&stack, &cfg).unwrap();
        assert_eq!(seg.k, 2, "stability should prefer the true k=2");
        assert!(two_regions_correct(&seg.labels, per));
    }

    #[test]
    fn parallel_matches_serial() {
        let per = 25;
        let stack = two_blob_stack(6, per, 0.05);
        let base = Config {
            n_neighbors: 8,
            n_components: 6,
            select: SelectSpec::Fixed(2),
            ..Config::default()
        };
        let par = segment(
            &stack,
            &Config {
                parallel: true,
                ..base
            },
        )
        .unwrap();
        let ser = segment(
            &stack,
            &Config {
                parallel: false,
                ..base
            },
        )
        .unwrap();
        assert_eq!(par.labels, ser.labels);
        assert_eq!(par.k, ser.k);
    }

    #[test]
    fn seed_stable() {
        let per = 25;
        let stack = two_blob_stack(6, per, 0.05);
        let cfg = Config {
            n_neighbors: 8,
            n_components: 6,
            select: SelectSpec::Fixed(3),
            ..Config::default()
        };
        let a = segment(&stack, &cfg).unwrap();
        let b = segment(&stack, &cfg).unwrap();
        assert_eq!(a.labels, b.labels);
    }

    /// A single *connected* chain of loci (the realistic case: one surface,
    /// graded regions) split into three contiguous bands. Spectral
    /// clustering should recover three contiguous segments; we allow a few
    /// boundary loci to slip.
    #[test]
    fn recovers_three_bands_connected() {
        let per = 20;
        let m = 3 * per;
        let d = 3;
        let n = 8;
        let mut rng = 0x1234_5678_u64;
        let nx = |s: &mut u64| -> f64 {
            *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = *s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        // Loci evenly along x in [0, 3); the graph connects adjacent loci, so
        // it is a single connected component.
        let mut flat = vec![0.0_f64; n * m * d];
        for i in 0..n {
            for p in 0..m {
                let x = p as f64 * (3.0 / m as f64);
                let base = i * m * d + p * d;
                flat[base] = x + 0.02 * nx(&mut rng);
                flat[base + 1] = 0.02 * nx(&mut rng);
                flat[base + 2] = 0.02 * nx(&mut rng);
            }
        }
        let stack = Stack::from_flat(&flat, n, m, d, None).unwrap();
        let cfg = Config {
            n_neighbors: 6,
            n_components: 8,
            select: SelectSpec::Fixed(3),
            ..Config::default()
        };
        let seg = segment(&stack, &cfg).unwrap();
        assert_eq!(seg.k, 3);
        // Three contiguous bands => three distinct labels, each dominating a
        // third of the chain. Count correctly-grouped loci by majority label
        // per true band.
        let mut correct = 0usize;
        for band in 0..3 {
            let slice = &seg.labels[band * per..(band + 1) * per];
            let mut counts = [0usize; 3];
            for &l in slice {
                counts[l] += 1;
            }
            correct += *counts.iter().max().unwrap();
        }
        assert!(
            correct as f64 / m as f64 >= 0.9,
            "only {correct}/{m} in-band"
        );
        // All three regions actually used.
        let distinct: std::collections::BTreeSet<_> = seg.labels.iter().collect();
        assert_eq!(distinct.len(), 3);
    }

    /// The parallel bootstrap-stability path must be bitwise-identical to the
    /// serial one: same k, same labels, same stability score.
    #[test]
    fn stability_parallel_matches_serial() {
        let per = 30;
        let stack = two_blob_stack(16, per, 0.05);
        let base = Config {
            n_neighbors: 8,
            n_components: 6,
            select: SelectSpec::Stability {
                max_k: 6,
                n_boot: 12,
                seed: 7,
            },
            ..Config::default()
        };
        let par = segment(
            &stack,
            &Config {
                parallel: true,
                ..base
            },
        )
        .unwrap();
        let ser = segment(
            &stack,
            &Config {
                parallel: false,
                ..base
            },
        )
        .unwrap();
        assert_eq!(par.k, ser.k);
        assert_eq!(par.labels, ser.labels);
        assert_eq!(par.stability, ser.stability);
    }

    #[test]
    fn embedding_shape_matches_requested_dimensions() {
        let stack = two_blob_stack(6, 30, 0.05);
        let seg = segment(
            &stack,
            &Config {
                n_neighbors: 8,
                n_components: 6,
                select: SelectSpec::Fixed(2),
                ..Config::default()
            },
        )
        .unwrap();
        assert_eq!(seg.embedding.shape(), (60, 6));
    }

    #[test]
    fn rejects_non_finite_stack_values() {
        let data = vec![0.0, 0.0, f64::NAN, 1.0, 1.0, 1.0];
        assert!(matches!(
            Stack::from_flat(&data, 1, 2, 3, None),
            Err(HdmError::InvalidStack(_))
        ));
    }

    #[test]
    fn rejects_invalid_configuration() {
        let stack = two_blob_stack(4, 10, 0.05);
        let invalid = [
            Config {
                n_neighbors: 0,
                ..Config::default()
            },
            Config {
                diffusion_time: f64::NAN,
                ..Config::default()
            },
            Config {
                select: SelectSpec::Fixed(0),
                ..Config::default()
            },
            Config {
                select: SelectSpec::Stability {
                    max_k: 4,
                    n_boot: 0,
                    seed: 0,
                },
                ..Config::default()
            },
        ];
        for cfg in invalid {
            assert!(matches!(
                segment(&stack, &cfg),
                Err(HdmError::InvalidConfig(_))
            ));
        }

        let too_many = Config {
            select: SelectSpec::Fixed(stack.m() + 1),
            ..Config::default()
        };
        assert!(matches!(
            segment(&stack, &too_many),
            Err(HdmError::TooManyClusters { .. })
        ));
    }

    #[test]
    fn consensus_rejects_bad_subsets() {
        let stack = two_blob_stack(4, 10, 0.05);
        let graph = build_graph(&stack, &Config::default());
        assert!(affinity::consensus(&stack, &graph, &[]).is_err());
        assert!(affinity::consensus(&stack, &graph, &[stack.n()]).is_err());
    }

    #[test]
    fn isolated_nodes_become_singletons_in_normalized_operator() {
        let normalized = diffusion::normalize(&DMatrix::zeros(3, 3));
        assert_eq!(normalized.degree.as_slice(), &[1.0, 1.0, 1.0]);
        assert_eq!(normalized.s, DMatrix::identity(3, 3));
    }

    #[test]
    fn kmeans_uses_every_requested_cluster_on_degenerate_features() {
        let labels = cluster::kmeans(&DMatrix::zeros(6, 3), 3, 0);
        let distinct: std::collections::BTreeSet<_> = labels.iter().copied().collect();
        assert_eq!(distinct.len(), 3);
    }

    #[test]
    fn eigengap_and_modularity_select_valid_partitions() {
        let stack = two_blob_stack(8, 30, 0.05);
        for select in [
            SelectSpec::Eigengap { max_k: 6 },
            SelectSpec::Modularity { max_k: 6 },
        ] {
            let seg = segment(
                &stack,
                &Config {
                    n_neighbors: 8,
                    n_components: 6,
                    select,
                    ..Config::default()
                },
            )
            .unwrap();
            assert!((2..=6).contains(&seg.k));
            assert_eq!(seg.labels.len(), stack.m());
        }
    }
}
