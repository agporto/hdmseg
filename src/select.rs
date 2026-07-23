//! Model-selection metrics: Newman-Girvan modularity, the eigengap
//! heuristic, and the adjusted Rand index used by stability selection.

use nalgebra::DMatrix;
use std::collections::HashMap;

/// Newman-Girvan modularity `Q` of a labeling on the weighted graph `w`.
pub fn modularity(w: &DMatrix<f64>, labels: &[usize]) -> f64 {
    let m = w.nrows();
    let mut degree = vec![0.0_f64; m];
    let mut m2 = 0.0;
    for p in 0..m {
        let d: f64 = w.row(p).sum();
        degree[p] = d;
        m2 += d;
    }
    if m2 <= 0.0 {
        return 0.0;
    }
    let k = labels.iter().copied().max().map(|x| x + 1).unwrap_or(0);
    let mut l_in = vec![0.0_f64; k];
    let mut d_tot = vec![0.0_f64; k];
    for p in 0..m {
        d_tot[labels[p]] += degree[p];
        for q in 0..m {
            if labels[p] == labels[q] {
                l_in[labels[p]] += w[(p, q)];
            }
        }
    }
    let mut q = 0.0;
    for c in 0..k {
        let a = l_in[c] / m2;
        let b = d_tot[c] / m2;
        q += a - b * b;
    }
    q
}

/// Eigengap heuristic. `eigenvalues` are the leading eigenvalues of the
/// symmetric normalized operator `S`, descending (index 0 is the trivial
/// top one). Returns the `k in [2, max_k]` with the largest gap
/// `v[k-1] - v[k]`.
pub fn eigengap(eigenvalues: &[f64], max_k: usize) -> usize {
    let n = eigenvalues.len();
    let hi = max_k.min(n.saturating_sub(1)).max(2);
    let mut best_k = 2;
    let mut best_gap = f64::NEG_INFINITY;
    for k in 2..=hi {
        let gap = eigenvalues[k - 1] - eigenvalues[k];
        if gap > best_gap {
            best_gap = gap;
            best_k = k;
        }
    }
    best_k
}

/// Adjusted Rand index between two labelings of the same length. Returns 1.0
/// for identical partitions (up to relabeling), ~0.0 for chance agreement.
pub fn adjusted_rand_index(a: &[usize], b: &[usize]) -> f64 {
    let n = a.len();
    if n == 0 {
        return 1.0;
    }
    let mut table: HashMap<(usize, usize), u64> = HashMap::new();
    let mut row: HashMap<usize, u64> = HashMap::new();
    let mut col: HashMap<usize, u64> = HashMap::new();
    for i in 0..n {
        *table.entry((a[i], b[i])).or_insert(0) += 1;
        *row.entry(a[i]).or_insert(0) += 1;
        *col.entry(b[i]).or_insert(0) += 1;
    }
    let comb2 = |x: u64| -> f64 { (x as f64) * (x as f64 - 1.0) / 2.0 };
    let sum_table: f64 = table.values().map(|&x| comb2(x)).sum();
    let sum_row: f64 = row.values().map(|&x| comb2(x)).sum();
    let sum_col: f64 = col.values().map(|&x| comb2(x)).sum();
    let total = comb2(n as u64);
    let expected = sum_row * sum_col / total;
    let max_index = 0.5 * (sum_row + sum_col);
    if (max_index - expected).abs() < 1e-12 {
        return 1.0;
    }
    (sum_table - expected) / (max_index - expected)
}
