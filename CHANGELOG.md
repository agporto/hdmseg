# Changelog

All notable changes to hdmseg are documented here. The project follows
[Semantic Versioning](https://semver.org).

## [0.1.0] — unreleased

Initial release.

### Added

- Correspondence-collapsed consensus-diffusion segmentation of a population of
  homologous shapes in dense correspondence (`(N, M, D)` stack → per-locus
  region labels).
- Consensus (frame-free) diffusion operator with per-specimen self-tuning
  Gaussian affinity on a fixed kNN reference graph.
- Symmetric normalized operator with an exact dense eigensolver.
- Spectral clustering (Ng-Jordan-Weiss) with deterministic k-means.
- Model selection: bootstrap **stability** (default), Newman-Girvan
  modularity, and eigengap.
- Python bindings (PyO3 + maturin, abi3-py39) exposing `hdmseg.segment`.
- Validated configuration and finite-input handling with recoverable numerical
  errors.
- Deterministic bootstrap sampling and fixed-order parallel score reduction.

### Notes

- The implementation is frame-free consensus diffusion plus spectral
  clustering; it is not a horizontal or hypoelliptic diffusion map.
- The dense operator uses `O(M²)` memory and the eigensolve uses `O(M³)` time.
