"""Minimal hdmseg demo: segment a synthetic population of two-region shapes.

Run after installing from the repository: ``python segment_demo.py``.
"""

import numpy as np

import hdmseg


def make_population(n=20, per=40, jitter=0.05, seed=0):
    """N specimens, each with 2*per corresponding loci forming two blobs."""
    rng = np.random.default_rng(seed)
    base = np.zeros((2 * per, 3))
    base[:per] = rng.uniform(-0.5, 0.5, (per, 3))
    base[per:] = rng.uniform(-0.5, 0.5, (per, 3))
    base[per:, 0] += 10.0  # second blob offset along x
    return np.stack([base + jitter * rng.uniform(-0.5, 0.5, base.shape) for _ in range(n)])


def main():
    X = make_population()  # (N, M, 3)
    seg = hdmseg.segment(X, n_neighbors=8, n_components=6, max_k=6, n_boot=10)

    print(seg)  # Segmentation(k=..., M=..., modularity=..., stability=...)
    print("regions found:", seg.k)
    print("labels (first 10):", seg.labels[:10])
    print("bootstrap stability:", seg.stability)
    print("leading eigenvalues:", np.round(seg.eigenvalues[:4], 4))


if __name__ == "__main__":
    main()
