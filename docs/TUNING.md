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

## Dense-solver scale

hdmseg currently performs a full dense eigendecomposition:

- operator memory is `O(M²)`;
- eigendecomposition time is `O(M³)`;
- stability selection repeats the solve `n_boot` times.

There is no automatic large-data approximation. Benchmark representative
inputs and monitor peak memory before increasing `M` or `n_boot`. Fixing `k`
avoids bootstrap/model-selection solves but does not change the dense
eigendecomposition cost of one segmentation.

## Determinism

Given the same input, seed, build, and machine, serial and parallel execution
are expected to produce the same partition and stability score. Floating-point
results may differ slightly across CPU architectures; compare partitions with
the adjusted Rand index when testing portability.

## Reference shape

`reference` must have shape `(M, D)` and contain only finite values. It controls
only graph topology. If omitted, the plain specimen-wise mean is used; input
shapes are therefore expected to have been aligned beforehand.
