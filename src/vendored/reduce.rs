//! Deterministic parallel map.
//!
//! rustcpd guarantees bitwise parallel==serial by keeping a fixed reduction
//! order. Here the only parallel work is an independent per-locus (or
//! per-specimen) computation whose results are written back in index order,
//! so an order-preserving `map` collect is already deterministic. This
//! wrapper documents that intent and gives one switch for parallelism.

use rayon::prelude::*;

/// Map `f` over `0..n`, returning results in index order. When `parallel`,
/// work is distributed across the rayon pool but the output order (and thus
/// any downstream accumulation) is identical to the serial path.
#[allow(dead_code)]
pub(crate) fn map_indexed<T, F>(n: usize, parallel: bool, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync + Send,
{
    if parallel {
        (0..n).into_par_iter().map(f).collect()
    } else {
        (0..n).map(f).collect()
    }
}
