"""hdmseg — population-consistent surface segmentation by
correspondence-collapsed diffusion maps.

Given a stack ``X`` of shape ``(N, M, D)`` — ``N`` specimens sharing ``M``
loci in dense correspondence, in ``D`` dimensions — :func:`segment` builds one
consensus diffusion operator on the loci, embeds it spectrally, and clusters
the embedding into regions coherent across the whole sample.

```python
import numpy as np, hdmseg
seg = hdmseg.segment(X)          # X: (N, M, D)
seg.labels                       # (M,) region id per locus
seg.k, seg.modularity, seg.stability
```
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _distribution_version

import numpy as np

from ._core import Segmentation, segment as _segment

try:
    __version__ = _distribution_version("hdmseg")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["segment", "Segmentation", "__version__"]


def segment(
    X,
    *,
    k=None,
    n_neighbors: int = 12,
    n_components: int = 20,
    diffusion_time: float = 1.0,
    select: str = "stability",
    max_k: int = 12,
    n_boot: int = 20,
    reference=None,
    seed: int = 0,
    parallel: bool = True,
) -> Segmentation:
    """Segment a correspondence stack.

    Parameters
    ----------
    X : array_like, shape (N, M, D)
        ``N`` specimens, ``M`` corresponding loci, ``D in {2, 3}`` dims.
    k : int, optional
        Fixed number of regions. If ``None`` (default), ``k`` is chosen by
        ``select``.
    n_neighbors : int
        Neighbors in the kNN reference graph.
    n_components : int
        Diffusion-embedding dimension.
    diffusion_time : float
        Diffusion time ``t`` (eigenvalue power in the coordinates).
    select : {"stability", "modularity", "eigengap"}
        Model-selection criterion when ``k is None``. ``"stability"``
        (bootstrap over specimens) is the default.
    max_k : int
        Upper bound on ``k`` for selection.
    n_boot : int
        Bootstrap resamples for ``select="stability"``.
    reference : array_like, shape (M, D), optional
        Reference shape for the graph topology. Defaults to the plain mean of
        the loci (the builder is assumed to pre-align the specimens).
    seed : int
        Seed for k-means / bootstrap.
    parallel : bool
        Use the internal thread pool (results are identical to serial).

    Returns
    -------
    Segmentation
        With attributes ``labels`` (M,), ``k``, ``embedding``,
        ``eigenvalues``, ``modularity``, and ``stability``.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim != 3:
        raise ValueError(f"X must be (N, M, D); got shape {X.shape}")
    n, m, d = X.shape
    flat = X.reshape(-1)

    ref = None
    if reference is not None:
        reference = np.ascontiguousarray(reference, dtype=np.float64)
        if reference.shape != (m, d):
            raise ValueError(
                f"reference must have shape {(m, d)}; got {reference.shape}"
            )
        ref = reference.reshape(-1)

    return _segment(
        flat, n, m, d,
        n_neighbors=n_neighbors,
        n_components=n_components,
        diffusion_time=diffusion_time,
        select=select,
        k=k,
        max_k=max_k,
        n_boot=n_boot,
        reference=ref,
        seed=seed,
        parallel=parallel,
    )
