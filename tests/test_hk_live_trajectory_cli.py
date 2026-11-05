"""The opt-in HK live diagnostic must not accidentally contact paid providers."""

from __future__ import annotations

import runpy
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



@pytest.mark.unit
def test_hk_preflight_only_warms_history_without_starting_llm(monkeypatch, tmp_path, capsys):
    import pandas as pd

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_hk_trajectory.py"
    module = runpy.run_path(str(script), run_name="hk_trajectory_test")
    calls = []

    def fake_load(symbol, date):
        calls.append((symbol, date))
        return pd.DataFrame({
            "Date": [pd.Timestamp("2026-11-04")],
            "Close": [105.0],
        })

    monkeypatch.setattr(module["hk"], "load_ohlcv", fake_load)
    monkeypatch.setattr(
        sys, "argv",
        [str(script), "--ticker", "9988.HK", "--date", "2026-11-04", "--preflight-only"],
    )
    monkeypatch.chdir(tmp_path)
    assert module["main"]() == 0
    assert calls == [("9988.HK", "2026-11-04")]
    assert "No LLM or Agent execution requested." in capsys.readouterr().out
    assert not list(tmp_path.rglob("trace.jsonl"))
