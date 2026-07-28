//! # hdmseg
//!
//! Population-consistent surface segmentation by **correspondence-collapsed
//! consensus diffusion maps**. Given a stack of `N` homologous shapes sharing
//! `M` loci in dense correspondence, hdmseg builds one `M × M` frame-free
//! consensus operator on the loci and applies spectral clustering without
//! materializing its structural zeros.
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
use rayon::prelude::*;
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
    Graph::knn(stack, cfg.n_neighbors.max(1), cfg.parallel)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SolverKind {
    Auto,
    Dense,
    Sparse,
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

/// Normalize a sparse consensus operator and compute its requested spectral
/// products. Auto mode retains the dense implementation as a compatibility
/// fallback whenever the partial sparse solve is inapplicable or fails its
/// residual checks.
fn embedding_from_weights(
    graph: &Graph,
    weights: &affinity::EdgeWeights,
    n_vectors: usize,
    coord_dims: usize,
    cfg: &Config,
    solver: SolverKind,
    parallel: bool,
) -> Result<(Embedding, bool)> {
    let norm = diffusion::normalize_edges(weights, graph);
    if solver != SolverKind::Dense && spectral::sparse_applicable(&norm, graph, n_vectors) {
        match spectral::sparse(
            &norm,
            graph,
            n_vectors,
            coord_dims,
            cfg.diffusion_time,
            parallel,
        ) {
            Ok(embedding) => return Ok((embedding, true)),
            Err(error) if solver == SolverKind::Sparse => return Err(error),
            Err(_) => {}
        }
    } else if solver == SolverKind::Sparse {
        return Err(HdmError::EigenFailed);
    }

    let dense_norm = norm.to_dense(graph);
    let embedding = spectral::dense(
        &dense_norm,
        n_vectors,
        coord_dims,
        cfg.diffusion_time,
        parallel,
    )?;
    Ok((embedding, false))
}

/// Segment the stack.
pub fn segment(stack: &Stack, cfg: &Config) -> Result<Segmentation> {
    segment_with_solver(stack, cfg, SolverKind::Auto)
}

fn segment_with_solver(stack: &Stack, cfg: &Config, solver: SolverKind) -> Result<Segmentation> {
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
    // Stability reuses every per-specimen affinity row for the full operator
    // and all bootstrap resamples. Other selection modes stream rows in
    // bounded batches and retain only the consensus edge vector.
    let affinity_cache = if matches!(cfg.select, SelectSpec::Stability { .. }) {
        Some(affinity::AffinityCache::build(stack, &graph, cfg.parallel)?)
    } else {
        None
    };
    let full = full_subset(stack.n());
    let w_full = match &affinity_cache {
        Some(cache) => cache.consensus(&graph, &full)?,
        None => affinity::consensus_edges(stack, &graph, &full, cfg.parallel)?,
    };
    let (emb_full, full_used_sparse) = embedding_from_weights(
        &graph,
        &w_full,
        n_vectors,
        coord_dims,
        cfg,
        solver,
        cfg.parallel,
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
                let q = select::modularity_edges(&graph, &w_full, &labels);
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
            let cache = affinity_cache
                .as_ref()
                .expect("stability always constructs an affinity cache");

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

            let score_one = |subset: &Vec<usize>,
                             solve: SolverKind,
                             inner_parallel: bool|
             -> Result<Vec<f64>> {
                let weights = cache.consensus(&graph, subset)?;
                let (emb_b, _) = embedding_from_weights(
                    &graph,
                    &weights,
                    n_vectors,
                    coord_dims,
                    cfg,
                    solve,
                    inner_parallel,
                )?;
                Ok((2..=hi)
                    .enumerate()
                    .map(|(ki, k)| {
                        let labels_b = cluster::kmeans(&emb_b.vectors, k, cfg.seed);
                        select::adjusted_rand_index(&full_labels[ki], &labels_b)
                    })
                    .collect())
            };

            // Sparse bootstrap solves are independent. Run them across the
            // rayon pool with sequential inner kernels to avoid
            // oversubscription. If any partial solve misses convergence, redo
            // the batch sequentially through Auto so the dense reference
            // fallback remains available and memory stays bounded.
            let per_boot = if cfg.parallel && full_used_sparse && solver != SolverKind::Dense {
                let sparse_attempt: Result<Vec<Vec<f64>>> = subsets
                    .par_iter()
                    .map(|subset| score_one(subset, SolverKind::Sparse, false))
                    .collect();
                match sparse_attempt {
                    Ok(scores) => scores,
                    Err(error) if solver == SolverKind::Sparse => return Err(error),
                    Err(_) => subsets
                        .iter()
                        .map(|subset| score_one(subset, SolverKind::Auto, cfg.parallel))
                        .collect::<Result<Vec<_>>>()?,
                }
            } else {
                subsets
                    .iter()
                    .map(|subset| score_one(subset, solver, cfg.parallel))
                    .collect::<Result<Vec<_>>>()?
            };

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

    let modularity = select::modularity_edges(&graph, &w_full, &labels);

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

    fn connected_curve_stack(n: usize, m: usize, jitter: f64) -> Stack {
        let d = 3;
        let mut rng = 0xCAFE_F00D_1234_5678_u64;
        let unit = |state: &mut u64| -> f64 {
            *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = *state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let mut flat = vec![0.0; n * m * d];
        for i in 0..n {
            for p in 0..m {
                let t = p as f64 / (m - 1) as f64;
                let base = (i * m + p) * d;
                flat[base] = 5.0 * t + jitter * unit(&mut rng);
                flat[base + 1] = (3.0 * std::f64::consts::TAU * t).sin() + jitter * unit(&mut rng);
                flat[base + 2] = (2.0 * std::f64::consts::TAU * t).cos() + jitter * unit(&mut rng);
            }
        }
        Stack::from_flat(&flat, n, m, d, None).unwrap()
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
        assert!(affinity::consensus_edges(&stack, &graph, &[], true).is_err());
        assert!(affinity::consensus_edges(&stack, &graph, &[stack.n()], true).is_err());
    }

    #[test]
    fn sparse_affinity_is_bitwise_identical_to_legacy_dense_affinity() {
        let stack = two_blob_stack(9, 40, 0.05);
        let graph = Graph::knn(&stack, 10, false);
        let subset = [8, 1, 1, 4, 0, 7, 3, 8, 2];
        let legacy = affinity::consensus(&stack, &graph, &subset).unwrap();
        let serial = affinity::consensus_edges(&stack, &graph, &subset, false)
            .unwrap()
            .to_dense(&graph);
        let parallel = affinity::consensus_edges(&stack, &graph, &subset, true)
            .unwrap()
            .to_dense(&graph);
        let cached = affinity::AffinityCache::build(&stack, &graph, true)
            .unwrap()
            .consensus(&graph, &subset)
            .unwrap()
            .to_dense(&graph);
        assert_eq!(serial, legacy);
        assert_eq!(parallel, legacy);
        assert_eq!(cached, legacy);
    }

    #[test]
    fn parallel_top_k_graph_matches_serial_graph() {
        let stack = two_blob_stack(5, 80, 0.03);
        let serial = Graph::knn(&stack, 12, false);
        let parallel = Graph::knn(&stack, 12, true);
        assert_eq!(serial.edge_slice(), parallel.edge_slice());
        for p in 0..stack.m() {
            assert_eq!(serial.neighbors(p), parallel.neighbors(p));
            assert_eq!(serial.incident_edges(p), parallel.incident_edges(p));
        }
    }

    #[test]
    fn sparse_normalization_and_modularity_match_dense_operator_bitwise() {
        let stack = two_blob_stack(7, 45, 0.04);
        let graph = Graph::knn(&stack, 11, false);
        let subset: Vec<_> = (0..stack.n()).collect();
        let edges = affinity::consensus_edges(&stack, &graph, &subset, true).unwrap();
        let dense_w = edges.to_dense(&graph);
        let dense_norm = diffusion::normalize(&dense_w);
        let sparse_norm = diffusion::normalize_edges(&edges, &graph).to_dense(&graph);
        assert_eq!(sparse_norm.degree, dense_norm.degree);
        // faer's self-adjoint eigensolver consumes the lower triangle.  The
        // sparse representation matches that triangle bit-for-bit.  The old
        // dense matrix can differ by one rounding bit across its unused upper
        // triangle because the two multiplication orders are reversed.
        for p in 0..stack.m() {
            for q in 0..=p {
                assert_eq!(sparse_norm.s[(p, q)], dense_norm.s[(p, q)]);
            }
        }

        let labels: Vec<usize> = (0..stack.m()).map(|p| (p / 13) % 5).collect();
        assert_eq!(
            select::modularity_edges(&graph, &edges, &labels),
            select::modularity(&dense_w, &labels)
        );
    }

    #[test]
    fn sparse_and_dense_eigenspaces_are_equivalent() {
        let stack = connected_curve_stack(6, 320, 0.002);
        let graph = Graph::knn(&stack, 10, true);
        let subset: Vec<_> = (0..stack.n()).collect();
        let weights = affinity::consensus_edges(&stack, &graph, &subset, true).unwrap();
        let normalized_edges = diffusion::normalize_edges(&weights, &graph);
        assert!(spectral::sparse_applicable(&normalized_edges, &graph, 10));
        let dense = spectral::dense(&normalized_edges.to_dense(&graph), 10, 8, 1.0, true).unwrap();
        let sparse = spectral::sparse(&normalized_edges, &graph, 10, 8, 1.0, true).unwrap();
        for j in 0..10 {
            assert!(
                (dense.eigenvalues[j] - sparse.eigenvalues[j]).abs() < 1e-8,
                "eigenvalue {j}: dense={}, sparse={}",
                dense.eigenvalues[j],
                sparse.eigenvalues[j]
            );
        }
        let dense_projector = &dense.vectors * dense.vectors.transpose();
        let sparse_projector = &sparse.vectors * sparse.vectors.transpose();
        let max_projector_error = dense_projector
            .iter()
            .zip(sparse_projector.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_projector_error < 1e-6,
            "projector error {max_projector_error}"
        );
    }

    #[test]
    fn sparse_pipeline_matches_dense_for_every_selection_mode() {
        let stack = connected_curve_stack(8, 320, 0.003);
        let selections = [
            SelectSpec::Fixed(4),
            SelectSpec::Eigengap { max_k: 5 },
            SelectSpec::Modularity { max_k: 5 },
            SelectSpec::Stability {
                max_k: 5,
                n_boot: 3,
                seed: 11,
            },
        ];
        for select in selections {
            let cfg = Config {
                n_neighbors: 10,
                n_components: 8,
                select,
                seed: 3,
                parallel: true,
                ..Config::default()
            };
            let dense = segment_with_solver(&stack, &cfg, SolverKind::Dense).unwrap();
            let sparse = segment_with_solver(&stack, &cfg, SolverKind::Sparse).unwrap();
            assert_eq!(sparse.k, dense.k, "selection mode {select:?}");
            assert!(
                (select::adjusted_rand_index(&sparse.labels, &dense.labels) - 1.0).abs() < 1e-12,
                "partition mismatch for {select:?}"
            );
            assert!(
                (sparse.modularity - dense.modularity).abs() < 1e-10,
                "modularity mismatch for {select:?}"
            );
            match (dense.stability, sparse.stability) {
                (Some(a), Some(b)) => assert!(
                    (a - b).abs() < 1e-10,
                    "stability mismatch for {select:?}: {a} vs {b}"
                ),
                (None, None) => {}
                values => panic!("stability presence mismatch for {select:?}: {values:?}"),
            }
            for (j, (&a, &b)) in dense
                .eigenvalues
                .iter()
                .zip(&sparse.eigenvalues)
                .enumerate()
            {
                assert!(
                    (a - b).abs() < 1e-8,
                    "eigenvalue {j} mismatch for {select:?}: {a} vs {b}"
                );
            }
            let dense_gram = &dense.embedding * dense.embedding.transpose();
            let sparse_gram = &sparse.embedding * sparse.embedding.transpose();
            let max_error = dense_gram
                .iter()
                .zip(sparse_gram.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                max_error < 1e-6,
                "embedding Gram error {max_error} for {select:?}"
            );
        }
    }

    #[test]
    fn auto_falls_back_to_dense_for_disconnected_large_graphs() {
        let stack = two_blob_stack(5, 160, 0.02);
        let cfg = Config {
            n_neighbors: 10,
            n_components: 8,
            select: SelectSpec::Fixed(2),
            parallel: true,
            ..Config::default()
        };
        let dense = segment_with_solver(&stack, &cfg, SolverKind::Dense).unwrap();
        let auto = segment_with_solver(&stack, &cfg, SolverKind::Auto).unwrap();
        assert_eq!(auto.labels, dense.labels);
        assert_eq!(auto.eigenvalues, dense.eigenvalues);
        assert_eq!(auto.embedding, dense.embedding);
        assert_eq!(auto.modularity, dense.modularity);
    }

    #[test]
    fn sparse_parallel_matches_serial_end_to_end() {
        let stack = connected_curve_stack(10, 320, 0.003);
        let base = Config {
            n_neighbors: 10,
            n_components: 8,
            select: SelectSpec::Stability {
                max_k: 5,
                n_boot: 4,
                seed: 17,
            },
            seed: 5,
            ..Config::default()
        };
        let parallel = segment(
            &stack,
            &Config {
                parallel: true,
                ..base
            },
        )
        .unwrap();
        let serial = segment(
            &stack,
            &Config {
                parallel: false,
                ..base
            },
        )
        .unwrap();
        assert_eq!(parallel.k, serial.k);
        assert_eq!(
            select::adjusted_rand_index(&parallel.labels, &serial.labels),
            1.0
        );
        assert_eq!(parallel.stability, serial.stability);
        assert_eq!(parallel.modularity, serial.modularity);
        for (&a, &b) in parallel.eigenvalues.iter().zip(&serial.eigenvalues) {
            assert!((a - b).abs() < 1e-10, "{a} vs {b}");
        }
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
