# hdmseg

Python bindings for population-consistent spectral segmentation by
correspondence-collapsed consensus diffusion.

```bash
python -m pip install "git+https://github.com/agporto/hdmseg.git#subdirectory=python"
```

```python
import hdmseg

segmentation = hdmseg.segment(X)  # X has shape (N, M, D)
segmentation.labels               # one region id per corresponding locus
```

The current implementation uses dense `M × M` operators and an exact dense
eigendecomposition. It is frame-free and is not a horizontal or hypoelliptic
diffusion-map implementation.

See the repository README and `docs/TUNING.md` for the method, complete API,
complexity limits, and development instructions.
