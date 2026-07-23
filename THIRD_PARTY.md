# Vendored-code provenance

The files under `src/vendored/` were derived from the `rustcpd` project by
Arthur Porto:

- `fastexp.rs`
- `linalg.rs`
- `reduce.rs`
- `spatial.rs`

They are redistributed as part of hdmseg under the BSD 2-Clause terms in
[`LICENSE`](LICENSE). The Cephes exponential coefficients in `fastexp.rs` are
from Stephen L. Moshier's Cephes library and are public domain.

Normal Cargo dependencies are not copied into this source tree; their own
license metadata remains authoritative.
