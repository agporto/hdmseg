#!/usr/bin/env python3
"""Independent NumPy/SciPy check of the hdmseg math.

Generates a fixed (N,M,D) stack, has the Rust `verify_dump` example compute
the consensus operator W and the leading eigenvalues of S, and recomputes both
from scratch here. Passes if W matches entrywise and the spectra agree.
"""
import subprocess, sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

N, M, D = 6, 60, 3
K = 8
NC = 10
FLOOR = 1e-12

def build_stack():
    rng = np.random.default_rng(0)
    # Two blobs (A near origin, B near x=10), jittered per specimen.
    per = M // 2
    base = np.zeros((M, D))
    base[:per] = rng.uniform(-0.5, 0.5, (per, D))
    base[per:] = rng.uniform(-0.5, 0.5, (M - per, D))
    base[per:, 0] += 10.0
    X = np.empty((N, M, D))
    for i in range(N):
        X[i] = base + 0.05 * rng.uniform(-0.5, 0.5, (M, D))
    return X

def knn_symmetric(reference, k):
    m = reference.shape[0]
    d2 = ((reference[:, None, :] - reference[None, :, :]) ** 2).sum(-1)
    nbrs = [set() for _ in range(m)]
    for p in range(m):
        order = sorted((d2[p, q], q) for q in range(m) if q != p)
        for _, q in order[:k]:
            nbrs[p].add(q)
            nbrs[q].add(p)
    return [sorted(s) for s in nbrs]

def consensus_W(X, nbrs):
    n, m, d = X.shape
    W = np.zeros((m, m))
    for i in range(n):
        x = X[i]
        # per-locus median-neighbor bandwidth
        sigma = np.empty(m)
        for p in range(m):
            ds = sorted(np.linalg.norm(x[p] - x[q]) for q in nbrs[p])
            sigma[p] = max(ds[len(ds) // 2], FLOOR) if ds else FLOOR
        for p in range(m):
            for q in nbrs[p]:
                if q > p:
                    dist2 = ((x[p] - x[q]) ** 2).sum()
                    val = np.exp(-dist2 / (sigma[p] * sigma[q]))
                    W[p, q] += val
                    W[q, p] += val
    return W / n

def spectrum(W, nc):
    deg = W.sum(1)
    inv = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    S = (inv[:, None] * W) * inv[None, :]
    vals = np.linalg.eigvalsh(S)  # ascending
    return np.sort(vals)[::-1][:nc]

def main():
    X = build_stack()
    stack_path = os.path.join(HERE, "stack.txt")
    w_path = os.path.join(HERE, "W_rust.txt")
    with open(stack_path, "w") as f:
        f.write(f"{N} {M} {D}\n")
        f.write(" ".join(f"{v:.17e}" for v in X.reshape(-1)))

    out = subprocess.run(
        ["cargo", "run", "--release", "--features", "verification",
         "--example", "verify_dump", "--",
         stack_path, w_path, str(K), str(NC)],
        cwd=ROOT, capture_output=True, text=True,
    )
    if out.returncode != 0:
        print(out.stderr); sys.exit(1)
    ev_rust = np.array([float(t) for t in out.stdout.split()])

    W_rust = np.loadtxt(w_path)
    nbrs = knn_symmetric(X.mean(0), K)
    W_ref = consensus_W(X, nbrs)

    w_err = np.abs(W_rust - W_ref).max()
    ev_ref = spectrum(W_ref, NC)
    ev_err = np.abs(np.sort(ev_rust)[::-1] - ev_ref).max()

    print(f"W max abs diff (rust vs numpy):        {w_err:.3e}")
    print(f"S eigenvalue max abs diff:             {ev_err:.3e}")
    print(f"leading eigenvalues (numpy): {np.round(ev_ref, 6)}")
    ok = w_err < 1e-9 and ev_err < 1e-8
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 2)

if __name__ == "__main__":
    main()
