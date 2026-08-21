#!/usr/bin/env python3
"""Validate that hdmseg release metadata agrees across the repository."""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGES = ("hdmseg", "hdmseg-python")


def cargo_version(path: Path) -> str:
    with path.open("rb") as stream:
        return str(tomllib.load(stream)["package"]["version"])


def citation_version(text: str) -> str | None:
    match = re.search(r"(?m)^version:\s*['\"]?([0-9A-Za-z.+-]+)['\"]?\s*$", text)
    return None if match is None else match.group(1)


def lock_package_version(lock_text: str, package: str) -> str | None:
    pattern = re.compile(
        rf"(?ms)^\[\[package\]\]\s*\nname = \"{re.escape(package)}\"\s*\n"
        r"version = \"([^\"]+)\""
    )
    match = pattern.search(lock_text)
    return None if match is None else match.group(1)


def validate(*, expected_tag: str | None, skip_lock: bool) -> str:
    version = cargo_version(ROOT / "Cargo.toml")
    binding_version = cargo_version(ROOT / "python" / "Cargo.toml")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    errors: list[str] = []

    if binding_version != version:
        errors.append(
            f"python/Cargo.toml is {binding_version}, expected {version}"
        )

    cff_version = citation_version(citation)
    if cff_version != version:
        errors.append(
            f"CITATION.cff is {cff_version or 'missing'}, expected {version}"
        )

    if f"## [{version}]" not in changelog:
        errors.append(f"CHANGELOG.md has no release section for {version}")

    if expected_tag is not None and expected_tag != f"v{version}":
        errors.append(
            f"release tag {expected_tag!r} does not match manifest version v{version}"
        )

    if not skip_lock:
        lock_text = (ROOT / "Cargo.lock").read_text(encoding="utf-8")
        for package in LOCAL_PACKAGES:
            locked = lock_package_version(lock_text, package)
            if locked != version:
                errors.append(
                    f"Cargo.lock records {package}={locked or 'missing'}, expected {version}"
                )

    if errors:
        raise ValueError("release metadata is inconsistent:\n- " + "\n- ".join(errors))
    return version


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-tag")
    parser.add_argument(
        "--skip-lock",
        action="store_true",
        help="Skip Cargo.lock validation before Cargo has synchronized it.",
    )
    parser.add_argument("--print-version", action="store_true")
    args = parser.parse_args()

    try:
        version = validate(
            expected_tag=args.expected_tag,
            skip_lock=bool(args.skip_lock),
        )
    except (OSError, KeyError, ValueError, tomllib.TOMLDecodeError) as error:
        print(error, file=sys.stderr)
        return 1

    if args.print_version:
        print(version)
    else:
        lock_scope = "excluding Cargo.lock" if args.skip_lock else "including Cargo.lock"
        print(f"release metadata verified for hdmseg {version} ({lock_scope})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
