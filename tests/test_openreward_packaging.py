"""Guards openreward: pin in pyproject must match a resolvable range on public PyPI."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


def _openreward_dependency_line() -> str:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    for dep in data["project"]["dependencies"]:
        d = dep.strip()
        if d.lower().startswith("openreward"):
            return d
    raise AssertionError("pyproject must list openreward in [project] dependencies")


def test_openreward_requirement_parses_with_minimum_0_1_89():
    line = _openreward_dependency_line()
    req = Requirement(line)
    assert req.name == "openreward"
    min_bound: Version | None = None
    for sp in req.specifier:
        if sp.operator in (">=", ">", "~="):
            min_bound = Version(sp.version)
            break
    assert min_bound is not None, f"add a lower bound, e.g. openreward>=0.1.89, got: {line!r}"
    assert min_bound >= Version("0.1.89")


@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("PYPI_CHECK", "") != "1",
    reason="set PYPI_CHECK=1 to run live PyPI check (network)",
)
def test_pypi_published_latest_satisfies_project_pin() -> None:
    """Catches environments where `pip` is so old the index only lists ancient wheels."""
    out = subprocess.run(
        [sys.executable, "-m", "pip", "index", "versions", "openreward"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    m = re.search(r"openreward \(([0-9][0-9.]*)\)", out)
    assert m, f"unexpected pip index output, got: {out[:400]!r}"
    latest = Version(m.group(1))
    line = _openreward_dependency_line()
    req = Requirement(line)
    assert req.specifier.contains(
        latest, prereleases=True
    ), f"latest PyPI openreward {latest} must satisfy {line!r}"
