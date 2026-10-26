"""CLI 无参数时仍执行常规分析，同时支持 `backtest` 子命令。

文档中的调用形式为 `tradingagents --checkpoint`，因此增加子命令后，
仍须保持分析功能为默认行为。"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

import cli.main as m


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.setattr(m, "run_analysis", lambda **kw: calls.append(("analysis", kw)))
    calls.clear()
    return CliRunner()


calls: list = []


@pytest.mark.unit
def test_no_arguments_still_runs_an_analysis(runner):
    assert runner.invoke(m.app, []).exit_code == 0
    no_flags = {"ticker": None, "date": None, "analysts": None, "save": None, "show": None, "html": None}
    assert calls == [("analysis", {"checkpoint": None, "portfolio": None, "flags": no_flags})]


@pytest.mark.unit
def test_options_still_parse_without_a_subcommand(runner):
    assert runner.invoke(m.app, ["--checkpoint"]).exit_code == 0
    assert calls[0][1]["checkpoint"] is True


@pytest.mark.unit
def test_backtest_does_not_also_run_an_analysis(runner, monkeypatch, tmp_path):
    swept = []
    monkeypatch.setattr(m, "run_backtest", lambda *a, **kw: swept.append((a, kw)) or _Result(tmp_path))
    monkeypatch.setattr(m, "summarize", lambda log: _Summary())

    result = runner.invoke(m.app, ["backtest", "NVDA,AAPL", "--start", "2026-06-01",
                                   "--end", "2026-06-15", "--every", "7"])

    assert result.exit_code == 0, result.output
    assert calls == []  # the interactive analysis must not run
    (tickers, dates, _config), kwargs = swept[0]
    assert tickers == ["NVDA", "AAPL"]
    assert dates == ["2026-06-01", "2026-06-08", "2026-06-15"]
    assert "scored" in result.output


@pytest.mark.unit
def test_backtest_reports_a_bad_date_instead_of_a_traceback(runner):
    result = runner.invoke(m.app, ["backtest", "NVDA", "--start", "June", "--end", "2026-06-15"])
    assert result.exit_code == 1
    assert "YYYY-MM-DD" in result.output


@pytest.mark.unit
def test_help_lists_the_backtest_command(runner):
    assert "backtest" in runner.invoke(m.app, ["--help"]).output


class _Result:
    def __init__(self, tmp_path):
        self.run_id = "20260916_000000"
        self.log_path = tmp_path / "trading_memory.md"
        self.cells_run = 2
        self.skipped = 0
        self.failures = []
        self.settlement_failures = []


class _Summary:
    def render(self):
        return "scored 2 cells"


@pytest.mark.unit
def test_every_command_is_registered_when_run_as_a_module():
    """README 使用 `python -m cli.main`，该方式从上到下运行文件。
    因此新命令需要在 __main__ 入口执行前注册。"""
    import re
    import subprocess
    import sys

    out = subprocess.run([sys.executable, "-m", "cli.main", "backtest", "--help"],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-400:]
    # 终端启用颜色后，帮助文本可能用转义字符切开 --start 等参数；
    # 检查选项前应先去除这些控制码。
    plain = re.sub(r"\x1b\[[0-9;]*m", "", out.stdout)
    assert "--start" in plain


@pytest.mark.unit
def test_backtest_can_continue_an_interrupted_sweep(runner, monkeypatch, tmp_path):
    """长时间批量回测需要从中断处恢复，Python API 也应支持此行为。"""
    swept = []
    monkeypatch.setattr(m, "run_backtest", lambda *a, **kw: swept.append(kw) or _Result(tmp_path))
    monkeypatch.setattr(m, "summarize", lambda log: _Summary())

    result = runner.invoke(m.app, ["backtest", "NVDA", "--start", "2026-06-01",
                                   "--end", "2026-06-08", "--run-id", "20260617_120000"])

    assert result.exit_code == 0, result.output
    assert swept[0]["run_id"] == "20260617_120000"


@pytest.mark.unit
@pytest.mark.parametrize("args, expected", [
    (["backtest", "NVDA", "--start", "2026-08-01", "--end", "2026-06-08"], "before"),
    (["backtest", ",,", "--start", "2026-06-01", "--end", "2026-06-08"], "ticker"),
])
def test_backtest_rejects_input_that_would_sweep_nothing(runner, args, expected):
    """日期范围颠倒或 ticker 列表为空不能返回表面成功的零任务结果，
    应明确告知用户输入无效。"""
    result = runner.invoke(m.app, args)
    assert result.exit_code == 1
    assert expected in result.output.lower()


@pytest.mark.unit
def test_backtest_reports_a_setup_failure_in_one_line(runner, monkeypatch):
    """缺少密钥或分析师名称错误时，应显示明确的使用提示，而非原始回溯。"""
    def _explode(*a, **kw):
        raise ValueError("API key for provider 'openai' is not set")

    monkeypatch.setattr(m, "run_backtest", _explode)
    result = runner.invoke(m.app, ["backtest", "NVDA", "--start", "2026-06-01", "--end", "2026-06-08"])

    assert result.exit_code == 1
    assert "API key" in result.output
    assert "Traceback" not in result.output


@pytest.mark.unit
def test_backtest_shows_each_cell_and_how_to_continue(runner, monkeypatch, tmp_path):
    def sweep(tickers, dates, config, progress=None, **kw):
        for i, date in enumerate(dates, 1):
            progress(i, len(dates), tickers[0], date)
        return _Result(tmp_path)

    monkeypatch.setattr(m, "run_backtest", sweep)
    monkeypatch.setattr(m, "summarize", lambda log: _Summary())

    result = runner.invoke(m.app, ["backtest", "NVDA", "--start", "2026-06-01", "--end", "2026-06-08"])

    assert "[1/2] NVDA 2026-06-01" in result.output and "[2/2] NVDA 2026-06-08" in result.output
    assert f"--run-id {_Result(tmp_path).run_id}" in result.output


@pytest.mark.unit
@pytest.mark.parametrize("args, analysts", [
    pytest.param(["--analysts", "sentiment,market"], ["market", "social"], id="named_as_in_analyze"),
    pytest.param(["--asset-type", "crypto"], ["market", "social", "news"], id="crypto_default"),
])
def test_backtest_takes_its_analysts_as_an_analysis_does(runner, monkeypatch, tmp_path, args, analysts):
    swept = []
    monkeypatch.setattr(m, "run_backtest", lambda *a, **kw: swept.append(kw) or _Result(tmp_path))
    monkeypatch.setattr(m, "summarize", lambda log: _Summary())

    result = runner.invoke(m.app, ["backtest", "NVDA", "--start", "2026-06-01", "--end", "2026-06-01", *args])

    assert result.exit_code == 0, result.output
    assert swept[0]["selected_analysts"] == analysts


@pytest.mark.unit
@pytest.mark.parametrize("args", [
    pytest.param(["--analysts", "fundamentals", "--asset-type", "crypto"], id="unavailable_for_crypto"),
    pytest.param(["--asset-type", "bond"], id="unknown_asset_type"),
])
def test_backtest_refuses_a_selection_before_the_sweep(runner, monkeypatch, args):
    monkeypatch.setattr(m, "run_backtest", lambda *a, **kw: pytest.fail("the sweep started"))

    result = runner.invoke(m.app, ["backtest", "NVDA", "--start", "2026-06-01", "--end", "2026-06-01", *args])

    assert result.exit_code == 1


@pytest.mark.unit
def test_login_and_logout_chatgpt_do_not_run_an_analysis(runner, monkeypatch):
    from tradingagents.llm_clients import chatgpt_auth

    monkeypatch.setattr(chatgpt_auth, "sign_in", lambda say: "a@example.com")
    monkeypatch.setattr(chatgpt_auth, "sign_out", lambda: "revoked")
    signed_in = runner.invoke(m.app, ["login", "chatgpt"])
    assert signed_in.exit_code == 0 and "Signed in to ChatGPT as a@example.com" in signed_in.output
    signed_out = runner.invoke(m.app, ["logout", "chatgpt"])
    assert signed_out.exit_code == 0 and "Signed out of ChatGPT" in signed_out.output
    assert runner.invoke(m.app, ["login", "claude"]).exit_code != 0
    assert calls == []


@pytest.mark.unit
def test_a_failed_login_exits_with_the_reason(runner, monkeypatch):
    from tradingagents.llm_clients import chatgpt_auth

    def refuse(say):
        raise chatgpt_auth.ChatGPTAuthError("Signing in with ChatGPT timed out")

    monkeypatch.setattr(chatgpt_auth, "sign_in", refuse)
    result = runner.invoke(m.app, ["login", "chatgpt"])
    assert result.exit_code == 1 and "timed out" in result.output


@pytest.mark.unit
@pytest.mark.parametrize("outcome, said", [
    (None, "Not signed in"), ("forgotten", "could not be reached to revoke"),
])
def test_logout_says_whether_openai_revoked_the_sign_in(runner, monkeypatch, outcome, said):
    from tradingagents.llm_clients import chatgpt_auth

    monkeypatch.setattr(chatgpt_auth, "sign_out", lambda: outcome)
    result = runner.invoke(m.app, ["logout", "chatgpt"])
    assert result.exit_code == 0 and said in " ".join(result.output.split())


@pytest.mark.unit
def test_a_failure_message_with_brackets_is_printed_as_text(runner, monkeypatch):
    from tradingagents.llm_clients import chatgpt_auth

    def refuse(say):
        raise chatgpt_auth.ChatGPTAuthError("ChatGPT sign-in failed ([/bogus]); sign in again.")

    monkeypatch.setattr(chatgpt_auth, "sign_in", refuse)
    result = runner.invoke(m.app, ["login", "chatgpt"])
    assert result.exit_code == 1 and "[/bogus]" in result.output
