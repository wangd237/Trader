"""The opt-in HK live diagnostic must not accidentally contact paid providers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.unit
def test_live_hk_trajectory_cli_requires_explicit_opt_in():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_hk_trajectory.py"
    probe = subprocess.run(
        [sys.executable, str(script), "--ticker", "9988.HK", "--date", "2026-11-04"],
        capture_output=True, text=True, check=False,
    )
    assert probe.returncode != 0
    assert "--confirm-live" in probe.stderr
    help_result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True, text=True, check=False,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "--analysts" in help_result.stdout
