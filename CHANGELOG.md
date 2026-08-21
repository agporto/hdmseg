# Changelog

All notable changes to hdmseg are documented here. The project follows
[Semantic Versioning](https://semver.org).

## [0.2.0] — 2026-08-20

### Added

- Edge-sparse storage for the consensus affinity and normalized diffusion
  operators, avoiding materialization of structural zeros.
- A partial self-adjoint Krylov-Schur eigensolver for the leading spectral
  modes of large connected graphs.
- Strict residual and orthogonality checks for partial eigenpairs, with an
  automatic dense compatibility fallback.
- Cached per-specimen edge affinities for bootstrap stability selection.
- Deterministic parallel bootstrap solves and fixed-order score reduction.
- Parallel top-k reference-graph construction and bounded-batch consensus
  accumulation.
- A reproducible large-stack benchmark in `examples/bench_scale.rs`.
- Sparse-versus-dense equivalence tests covering fixed-k, eigengap,
  modularity, and stability selection.

### Changed

- Connected positive-weight graphs with at least 256 loci now use the sparse
  path automatically when the requested eigenspace is sufficiently smaller
  than the graph.
- Small graphs, disconnected graphs, and partial solves that do not pass the
  numerical checks continue to use the original exact dense implementation.
- Bootstrap stability reuses its affinity cache across resamples rather than
  recomputing specimen geometry for every solve.
- Documentation now describes sparse scaling, dense fallback behavior, and
  large-data tuning.

### Compatibility

- The public Rust and Python segmentation APIs are unchanged.
- The consensus-affinity definition, normalization, clustering method, model
  selection criteria, and deterministic seed behavior are unchanged.
- Sparse and dense paths are tested to produce equivalent eigenspaces and
  partitions within strict numerical tolerances.

## [0.1.0] — 2026-07-24

Initial release.

### Added

- Correspondence-collapsed consensus-diffusion segmentation of a population of
  homologous shapes in dense correspondence (`(N, M, D)` stack → per-locus
  region labels).
- Consensus frame-free diffusion operator with per-specimen self-tuning
  Gaussian affinity on a fixed kNN reference graph.
- Symmetric normalized operator with an exact dense eigensolver.
- Spectral clustering using the Ng-Jordan-Weiss construction and deterministic
  k-means.
- Model selection by bootstrap stability, Newman-Girvan modularity, or
  eigengap.
- Python bindings using PyO3 and maturin with the `abi3-py39` stable ABI.
- Validated configuration and finite-input handling with recoverable numerical
  errors.
- Deterministic bootstrap sampling and fixed-order parallel score reduction.

### Notes

- The implementation is frame-free consensus diffusion plus spectral
  clustering; it is not a horizontal or hypoelliptic diffusion map.
- Version 0.1.0 used dense `O(M²)` operator storage and an `O(M³)` full
  eigendecomposition.
