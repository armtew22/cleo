"""The api package must not import parcellation/mesh/inference directly.

Static import-graph check via import-linter.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
def test_importlinter_clean():
    repo = Path(__file__).resolve().parents[2]
    lint_imports = shutil.which("lint-imports") or str(Path(sys.executable).with_name("lint-imports"))
    res = subprocess.run(
        [lint_imports, "--config", str(repo / "pyproject.toml")],
        capture_output=True, text=True, cwd=str(repo),
    )
    assert res.returncode == 0, (
        f"import-linter failed:\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    )
