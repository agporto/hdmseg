# hdmseg

Python bindings for population-consistent spectral segmentation by
correspondence-collapsed consensus diffusion.

```bash
python -m pip install "hdmseg>=0.2,<0.3"
```

```python
import hdmseg

segmentation = hdmseg.segment(X)  # X has shape (N, M, D)
segmentation.labels               # one region id per corresponding locus
```

The current implementation uses edge-sparse operators and a partial
self-adjoint eigensolver for large connected graphs, with an exact dense
fallback for small, disconnected, or nonconverged cases. It is frame-free and
is not a horizontal or hypoelliptic diffusion-map implementation.

See the repository README and `docs/TUNING.md` for the method, complete API,
complexity limits, source installation, and development instructions.
