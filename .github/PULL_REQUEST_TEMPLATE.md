## Summary

What this PR changes and why.

## Checklist

- [ ] `cargo fmt --all` and `cargo clippy -p hdmseg --all-targets` are clean
- [ ] Tests added or updated (Rust in `src/`, Python in `python/tests/`)
- [ ] `cargo test -p hdmseg --release` and `python -m pytest python/tests` pass
- [ ] Determinism preserved (parallel and serial partitions/scores agree)
- [ ] Public API changes noted in `CHANGELOG.md`; breaking changes flagged
- [ ] Docs/docstrings updated if behavior or parameters changed

## Notes for reviewers

Anything specific to look at — tricky algebra, a reference to validate
against, performance implications.
