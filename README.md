# hdmseg

[![Wheels](https://github.com/agporto/hdmseg/actions/workflows/wheels.yml/badge.svg)](https://github.com/agporto/hdmseg/actions/workflows/wheels.yml)
[![Rust](https://github.com/agporto/hdmseg/actions/workflows/rust.yml/badge.svg)](https://github.com/agporto/hdmseg/actions/workflows/rust.yml)
[![License: BSD-2-Clause](https://img.shields.io/badge/License-BSD_2--Clause-blue.svg)](LICENSE)

**Population-consistent spectral segmentation by correspondence-collapsed
consensus diffusion**, with a deterministic Rust core and Python bindings.

Given `N` homologous shapes that share `M` loci in dense correspondence,
hdmseg constructs one frame-free diffusion operator on the loci. Spectral
clustering of that operator produces one partition shared by the population.

## Status

hdmseg is pre-release research software. The current implementation uses an
exact dense eigendecomposition. It is suitable only when `M × M` dense storage
and `O(M³)` eigendecomposition are practical; benchmark your intended locus
count before committing to the method.

The implementation is inspired by correspondence-aware diffusion methods, but
it is **not** a horizontal or hypoelliptic diffusion-map implementation. It has
no fibre-bundle state, horizontal transport, or hypoelliptic graph Laplacian.

## Install from GitHub

Python 3.9 or newer and a Rust 1.87 or newer toolchain are required when
building from source.

```bash
python -m pip install "git+https://github.com/agporto/hdmseg.git#subdirectory=python"
```

For local development:

```bash
git clone https://github.com/agporto/hdmseg.git
cd hdmseg
python -m pip install ./python
```

Tagged GitHub Releases contain prebuilt wheels and a source distribution.
The project is not published to PyPI or crates.io.

## Python example

```python
import hdmseg

# X: (N, M, D), with D in {2, 3}
seg = hdmseg.segment(
    X,
    k=None,                  # fixed integer, or None for model selection
    n_neighbors=12,
    n_components=20,
    diffusion_time=1.0,
    select="stability",      # "stability" | "modularity" | "eigengap"
    max_k=12,
    n_boot=20,
    reference=None,          # optional (M, D) topology reference
    seed=0,
    parallel=True,
)

seg.labels        # (M,) region id per locus
seg.k             # chosen number of regions
seg.embedding     # (M, min(n_components, M - 1))
seg.eigenvalues   # leading normalized-operator eigenvalues
seg.modularity
seg.stability     # bootstrap stability, or None
```

Parameter guidance is in [`docs/TUNING.md`](docs/TUNING.md).

## Method

1. **Reference graph** — construct a symmetric union-kNN graph on the supplied
   reference shape or the specimen-wise mean.
2. **Consensus affinity** — on each specimen, evaluate Gaussian affinities on
   that fixed topology and average them. The local bandwidth is the median
   distance to graph neighbors in that specimen. This is a project-specific
   robust variant of self-tuning affinity, not the k-th-neighbor rule of
   Zelnik-Manor and Perona.
3. **Symmetric normalization** — form
   `S = D^{-1/2} W D^{-1/2}`. A zero-degree locus receives a unit self-loop and
   therefore remains an explicit singleton.
4. **Dense spectral solve** — compute the symmetric eigendecomposition of `S`.
5. **Clustering** — row-normalize the leading `k` eigenvectors and apply
   deterministic k-means (Ng-Jordan-Weiss).
6. **Model selection** — choose `k` by bootstrap stability, modularity, or an
   eigengap heuristic, unless a fixed `k` is supplied.

The exported diffusion coordinates use `|λ|^t D^{-1/2}u` and omit the trivial
leading mode. Clustering uses raw leading eigenvectors and is independent of
`diffusion_time`.

## Complexity and limits

- The brute-force reference kNN search is `O(M² log M)`.
- The consensus and normalized operators use `O(M²)` memory.
- The exact dense eigendecomposition is `O(M³)`.
- Bootstrap stability repeats operator construction and eigendecomposition
  `n_boot` times. Replicates run sequentially to bound dense memory; each
  eigensolve may use internal parallelism.

No sparse or approximate eigensolver is currently provided. Parallel and
serial execution are tested to produce the same partition and stability score
on a given machine. Exact floating-point identity across CPU architectures is
not guaranteed.

## Rust

```toml
[dependencies]
hdmseg = { git = "https://github.com/agporto/hdmseg" }
```

```rust
use hdmseg::{segment, Config, Stack};

let stack = Stack::from_flat(&flat, n, m, 3, None)?;
let segmentation = segment(&stack, &Config::default())?;
let labels = segmentation.labels;
```

## Propagating labels to a mesh

hdmseg labels corresponding loci. If the target mesh is denser, propagate
labels downstream using an application-appropriate method such as nearest
locus, geodesic assignment, or harmonic interpolation.

## References

- T. Gao, *Hypoelliptic Diffusion Maps I: Tangent Bundles*, 2015,
  arXiv:1503.05459.
- T. Gao, *The Diffusion Geometry of Fibre Bundles: Horizontal Diffusion
  Maps*, arXiv:1602.02330.
- R. Coifman and S. Lafon, *Diffusion maps*, Applied and Computational
  Harmonic Analysis, 2006.
- A. Ng, M. Jordan, and Y. Weiss, *On Spectral Clustering: Analysis and an
  Algorithm*, NIPS 2002.
- L. Zelnik-Manor and P. Perona, *Self-Tuning Spectral Clustering*, NIPS 2004.

## Development

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test -p hdmseg --release --all-features
python verify/check.py

cd python
python -m pip install maturin numpy pytest
maturin develop --release
python -m pytest tests -q
```

## License

BSD 2-Clause. See [`LICENSE`](LICENSE). Vendored-code provenance is recorded in
[`THIRD_PARTY.md`](THIRD_PARTY.md).
