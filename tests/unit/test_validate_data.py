from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_prepare_and_validate(tmp_path: Path):
    root = Path(__file__).resolve().parents[2]
    data = tmp_path / "data"
    prep = subprocess.run(
        [sys.executable, str(root / "training" / "prepare_data.py"), "--out-dir", str(data)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert prep.returncode == 0, prep.stderr
    val = subprocess.run(
        [sys.executable, str(root / "training" / "validate_data.py"), "--data-dir", str(data)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert val.returncode == 0, val.stdout + val.stderr
