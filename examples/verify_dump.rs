//! Verification helper: read a flat `(N,M,D)` stack from a text file, build
//! the consensus operator `W` and the normalized spectrum, dump `W` to a
//! file, and print the leading eigenvalues of `S`. A NumPy/SciPy reference
//! recomputes the same quantities to cross-check the math.
//!
//! Usage: `verify_dump <stack.txt> <W_out.txt> <k> <n_components>`
//! Input format: first line `N M D`, then `N*M*D` whitespace-separated f64
//! in row-major `(i, p, c)` order.

use hdmseg::Stack;
use hdmseg::verification;
use std::fs;
use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let text = fs::read_to_string(&args[1]).unwrap();
    let mut it = text.split_whitespace();
    let n: usize = it.next().unwrap().parse().unwrap();
    let m: usize = it.next().unwrap().parse().unwrap();
    let d: usize = it.next().unwrap().parse().unwrap();
    let data: Vec<f64> = it.map(|t| t.parse().unwrap()).collect();
    assert_eq!(data.len(), n * m * d);

    let k: usize = args[3].parse().unwrap();
    let n_components: usize = args[4].parse().unwrap();

    let stack = Stack::from_flat(&data, n, m, d, None).unwrap();
    let output = verification::operator_and_spectrum(&stack, k, n_components).unwrap();

    // Dump W.
    let mut f = fs::File::create(&args[2]).unwrap();
    for p in 0..m {
        let row: Vec<String> = (0..m)
            .map(|q| format!("{:.17e}", output.operator[(p, q)]))
            .collect();
        writeln!(f, "{}", row.join(" ")).unwrap();
    }

    // Print leading eigenvalues of S.
    let evs: Vec<String> = output
        .eigenvalues
        .iter()
        .map(|v| format!("{v:.17e}"))
        .collect();
    println!("{}", evs.join(" "));
}
