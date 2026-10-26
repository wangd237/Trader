"""包版本号只有一个权威来源：构建系统读取的 tradingagents.__version__。"""

import re
from pathlib import Path

import pytest

import tradingagents

PYPROJECT = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")


@pytest.mark.unit
def test_the_build_reads_the_version_from_the_package():
    assert re.search(r'^dynamic = \["version"\]', PYPROJECT, re.M)
    assert 'version = {attr = "tradingagents.__version__"}' in PYPROJECT
    assert not re.search(r'^version = "', PYPROJECT, re.M)
    assert re.fullmatch(r"\d+\.\d+\.\d+(\.dev\d+|rc\d+)?", tradingagents.__version__)
