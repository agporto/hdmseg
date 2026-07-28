//! Reproducible large-stack smoke benchmark.
//!
//! Usage:
//! `cargo run --release --example bench_scale -- [N] [M] [mode] [n_boot]`
//! where mode is `fixed`, `eigengap`, `modularity`, or `stability`.

use hdmseg::{Config, SelectSpec, Stack, segment};
use std::time::Instant;

fn argument<T: std::str::FromStr>(index: usize, default: T) -> T {
    std::env::args()
        .nth(index)
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn label_hash(labels: &[usize]) -> u64 {
    labels.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &label| {
        (hash ^ label as u64).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn main() {
    let n = argument(1, 1_000usize);
    let m = argument(2, 5_000usize);
    let mode = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "fixed".to_owned());
    let n_boot = argument(4, 3usize);
    let d = 3;

    // A smoothly deformed Fibonacci sphere: connected at k=12, deterministic,
    // and large enough to exercise every production kernel.
    let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    let mut base = vec![[0.0_f64; 3]; m];
    let mut harmonic = vec![0.0_f64; m];
    for p in 0..m {
        let y = 1.0 - 2.0 * (p as f64 + 0.5) / m as f64;
        let radius = (1.0 - y * y).sqrt();
        let theta = golden_angle * p as f64;
        base[p] = [radius * theta.cos(), y, radius * theta.sin()];
        harmonic[p] = (7.0 * theta).sin() * (1.0 - y * y);
    }
    let mut flat = vec![0.0_f64; n * m * d];
    for i in 0..n {
        let phase = i as f64 * 0.013;
        let amplitude = 0.015 * phase.sin();
        for p in 0..m {
            let scale = 1.0 + amplitude * harmonic[p];
            let offset = (i * m + p) * d;
            flat[offset] = scale * base[p][0];
            flat[offset + 1] = scale * base[p][1];
            flat[offset + 2] = scale * base[p][2];
        }
    }
    let stack = Stack::from_flat(&flat, n, m, d, None).expect("valid benchmark stack");
    drop(flat);

    let select = match mode.as_str() {
        "fixed" => SelectSpec::Fixed(6),
        "eigengap" => SelectSpec::Eigengap { max_k: 12 },
        "modularity" => SelectSpec::Modularity { max_k: 12 },
        "stability" => SelectSpec::Stability {
            max_k: 12,
            n_boot,
            seed: 0,
        },
        value => {
            panic!("mode must be fixed, eigengap, modularity, or stability, got {value}")
        }
    };
    let cfg = Config {
        n_neighbors: 12,
        n_components: 20,
        select,
        parallel: true,
        ..Config::default()
    };

    let started = Instant::now();
    let result = segment(&stack, &cfg).expect("segmentation succeeds");
    println!(
        "N={n} M={m} mode={mode} elapsed={:.3}s k={} label_hash={:016x} \
         modularity={:.6} stability={:?}",
        started.elapsed().as_secs_f64(),
        result.k,
        label_hash(&result.labels),
        result.modularity,
        result.stability
    );
}
