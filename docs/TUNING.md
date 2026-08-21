# Tuning hdmseg

All parameters below are keyword arguments to `hdmseg.segment`.

## Number of regions

- Pass `k=<integer>` when the number of regions is known.
- `select="stability"` resamples specimens and chooses the partition with the
  highest mean adjusted Rand index against the full-sample partition. It is
  the default and is most useful with roughly 15–20 or more specimens.
- `select="modularity"` maximizes Newman-Girvan modularity and avoids bootstrap
  solves, but can merge smaller regions.
- `select="eigengap"` selects the largest tested gap in the leading spectrum.
- `max_k` bounds automatic selection and must be at least 2.
- `n_boot` controls stability resamples and must be at least 1.

## Neighborhood

`n_neighbors` controls the symmetric union-kNN graph on the reference loci and
must be at least 1. Larger values make the operator smoother and more likely to
be connected. Smaller values sharpen local structure but can create separate
components.

The Gaussian bandwidth at each locus is the median distance to its graph
neighbors in each specimen. This is a project-specific robust local bandwidth,
not the k-th-nearest-neighbor rule from the original self-tuning spectral
clustering paper.

## Embedding

- `n_components` is the requested number of non-trivial diffusion coordinates.
  The returned width is `min(n_components, M - 1)`.
- Clustering uses the leading `k` raw normalized-operator eigenvectors, so
  `diffusion_time` does not affect labels.
- `diffusion_time` must be finite and non-negative. It controls the `|λ|^t`
  scaling of exported coordinates.

## Large-data scale

For a connected positive-weight graph with at least 256 loci, hdmseg stores
only the union-kNN edges and computes the requested leading eigenpairs with a
partial self-adjoint Krylov-Schur solve. The operator math is unchanged.

- Let `E` be the number of undirected graph edges. Operator storage is `O(E)`,
  and sparse matrix-vector products are `O(E)`.
- The eigensolver also stores an `M × r` Krylov basis, where `r` grows with
  the number of requested leading modes.
- `n_components` and `max_k` both affect how many leading modes are required.
- Stability selection caches `N × E` affinities once, then reuses them for all
  `n_boot` resamples. Larger `n_boot` adds solves but not affinity-cache size.
- Fixing `k`, eigengap selection, and modularity selection do not allocate the
  stability cache.

Small graphs use the dense solver because it is faster there. Disconnected
graphs and sparse solves that miss strict convergence checks also fall back to
the original dense implementation, which uses `O(M²)` memory and `O(M³)` time.
For a large unexpected fallback, first check whether `n_neighbors` produced a
disconnected graph; change it only when a denser neighborhood is scientifically
appropriate.

Benchmark representative inputs and monitor peak memory. A reproducible smoke
benchmark is available:

```bash
cargo run --release --example bench_scale -- 1000 5000 fixed
```

## Determinism

Given the same input, seed, build, and machine, serial and parallel execution
are expected to produce the same partition and stability score. Floating-point
results may differ slightly across CPU architectures; compare partitions with
the adjusted Rand index when testing portability.

## Reference shape

`reference` must have shape `(M, D)` and contain only finite values. It controls
only graph topology. If omitted, the plain specimen-wise mean is used; input
shapes are therefore expected to have been aligned beforehand.
