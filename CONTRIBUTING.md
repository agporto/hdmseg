# Contributing to hdmseg

Thanks for your interest in improving hdmseg. Bug reports, feature requests,
and pull requests are all welcome.

## Development setup

The project is a Cargo workspace: the Rust core is the root crate (`src/`),
the Python bindings (PyO3 + maturin) live in `python/`.

```bash
# Rust core
cargo test -p hdmseg --release
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all -- --check

# Python bindings
cd python
pip install maturin numpy pytest
maturin develop --release                           # build + install into the current env
python -m pytest tests -q
```

An independent NumPy cross-check of the numerical core lives in
`verify/check.py`.

## Before opening a pull request

- `cargo fmt --all` and workspace Clippy with all features/targets are clean.
- New behavior has tests (Rust in `src/`, Python in `python/tests/`). The
  numerical core is expected to stay deterministic — parallel and serial
  execution must produce the same partition and stability score on a given
  machine.
- Public API changes are noted in `CHANGELOG.md` under the unreleased
  heading, and breaking changes are called out explicitly (the project
  follows [Semantic Versioning](https://semver.org)).
- Algorithm changes that affect results are validated against a reference
  (e.g. extend `verify/check.py`).

## Releasing (maintainers)

The version is set in the root `Cargo.toml` and `python/Cargo.toml`; the
Python package reads it dynamically. Tagging a commit `vX.Y.Z` triggers the
`Wheels` workflow, which builds and tests abi3 wheels for Linux, macOS, and
Windows, builds a source distribution, and attaches the artifacts to a GitHub
Release. The workflow does not publish to PyPI.

## License

By contributing you agree that your contributions are licensed under the
project's [BSD 2-Clause License](LICENSE).
