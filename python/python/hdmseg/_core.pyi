"""Type stubs for the compiled ``hdmseg._core`` extension."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

class Segmentation:
    """Result of a segmentation."""

    @property
    def labels(self) -> npt.NDArray[np.int64]: ...
    @property
    def k(self) -> int: ...
    @property
    def embedding(self) -> npt.NDArray[np.float64]: ...
    @property
    def eigenvalues(self) -> npt.NDArray[np.float64]: ...
    @property
    def modularity(self) -> float: ...
    @property
    def stability(self) -> float | None: ...
    def __repr__(self) -> str: ...

def segment(
    data: npt.NDArray[np.float64],
    n: int,
    m: int,
    d: int,
    n_neighbors: int = ...,
    n_components: int = ...,
    diffusion_time: float = ...,
    select: str = ...,
    k: int | None = ...,
    max_k: int = ...,
    n_boot: int = ...,
    reference: npt.NDArray[np.float64] | None = ...,
    seed: int = ...,
    parallel: bool = ...,
) -> Segmentation: ...
