"""End-to-end tests for the hdmseg Python bindings."""

import numpy as np
import pytest

import hdmseg


def two_blob_stack(per=30, n=12, jitter=0.05, seed=0):
    rng = np.random.default_rng(seed)
    d = 3
    base = np.zeros((2 * per, d))
    base[:per] = rng.uniform(-0.5, 0.5, (per, d))
    base[per:] = rng.uniform(-0.5, 0.5, (per, d))
    base[per:, 0] += 10.0
    return np.stack([base + jitter * rng.uniform(-0.5, 0.5, (2 * per, d)) for _ in range(n)])


def blobs_recovered(labels, per):
    return (
        len(set(labels[:per])) == 1
        and len(set(labels[per:])) == 1
        and labels[0] != labels[per]
    )


def test_stability_recovers_two_blobs():
    X = two_blob_stack()
    seg = hdmseg.segment(X, n_neighbors=8, n_components=6, max_k=6, n_boot=10)
    assert seg.k == 2
    assert blobs_recovered(seg.labels, 30)
    assert seg.labels.dtype == np.int64
    assert seg.labels.shape == (60,)
    assert seg.stability == pytest.approx(1.0)


def test_fixed_k():
    X = two_blob_stack()
    seg = hdmseg.segment(X, k=2, n_neighbors=8, n_components=6)
    assert seg.k == 2
    assert seg.stability is None
    assert blobs_recovered(seg.labels, 30)


def test_deterministic():
    X = two_blob_stack()
    a = hdmseg.segment(X, n_neighbors=8, n_components=6, max_k=6, n_boot=10)
    b = hdmseg.segment(X, n_neighbors=8, n_components=6, max_k=6, n_boot=10)
    assert np.array_equal(a.labels, b.labels)


def test_parallel_matches_serial():
    X = two_blob_stack()
    a = hdmseg.segment(X, k=3, n_neighbors=8, n_components=6, parallel=True)
    b = hdmseg.segment(X, k=3, n_neighbors=8, n_components=6, parallel=False)
    assert np.array_equal(a.labels, b.labels)


def test_embedding_and_eigenvalues():
    X = two_blob_stack()
    seg = hdmseg.segment(X, k=2, n_neighbors=8, n_components=6)
    assert seg.embedding.shape == (60, 6)
    # Two disconnected blobs => two eigenvalue-1 components.
    assert seg.eigenvalues[0] == pytest.approx(1.0, abs=1e-9)
    assert seg.eigenvalues[1] == pytest.approx(1.0, abs=1e-9)


def test_reference_override():
    X = two_blob_stack()
    ref = X.mean(0)
    seg = hdmseg.segment(X, k=2, n_neighbors=8, n_components=6, reference=ref)
    assert blobs_recovered(seg.labels, 30)


def test_invalid_shape():
    with pytest.raises(ValueError):
        hdmseg.segment(np.zeros((3, 4)))


def test_invalid_select():
    X = two_blob_stack(per=10, n=4)
    with pytest.raises(ValueError):
        hdmseg.segment(X, select="bogus", n_neighbors=5, n_components=4)


def test_invalid_config_and_non_finite_values():
    X = two_blob_stack(per=10, n=4)
    with pytest.raises(ValueError):
        hdmseg.segment(X, n_neighbors=0)
    with pytest.raises(ValueError):
        hdmseg.segment(X, k=0)
    X[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        hdmseg.segment(X, k=2)


def test_invalid_reference_shape():
    X = two_blob_stack(per=10, n=4)
    with pytest.raises(ValueError):
        hdmseg.segment(X, k=2, reference=np.zeros((X.shape[1], 2)))
