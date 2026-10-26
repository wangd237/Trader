"""终端缺少控制台缓冲区时应输出一行可操作的错误提示（#1138）。"""
from __future__ import annotations

import sys

from typer.testing import CliRunner

import cli.main as m


def test_no_console_error_tuple_matches_platform():
    # 非 Windows 平台不能导入 win32 模块（模块会检查系统平台），
    # 因此异常类型元组应为空；except 接受空元组且不会匹配异常。
    # Windows 平台则使用真实异常类型，以便 prompt_toolkit 依赖故障
    # 在导入时明确报错，而不是悄然禁用处理。
    assert isinstance(m._NO_CONSOLE_ERRORS, tuple)
    assert all(issubclass(e, BaseException) for e in m._NO_CONSOLE_ERRORS)
    if sys.platform == "win32":
        assert m._NO_CONSOLE_ERRORS, "Windows must resolve the console error type"
    else:
        assert m._NO_CONSOLE_ERRORS == ()


def test_missing_console_prints_actionable_message(monkeypatch):
    class _NoConsole(Exception):
        pass

    # 通过替代类型在任意系统上模拟 Windows 的故障。
    monkeypatch.setattr(m, "_NO_CONSOLE_ERRORS", (_NoConsole,))

    def _boom(*a, **k):
        raise _NoConsole("No Windows console found. Are you running cmd.exe?")

    monkeypatch.setattr(m, "run_analysis", _boom)

    result = CliRunner().invoke(m.app, [])
    assert result.exit_code == 1
    assert "no Windows console available" in result.output
    # 原始 prompt_toolkit 异常堆栈不能直接展示给用户。
    assert "Traceback" not in result.output


def test_unrelated_errors_still_propagate(monkeypatch):
    # 异常处理应保持精准，只转换特定的控制台错误。
    monkeypatch.setattr(m, "_NO_CONSOLE_ERRORS", (RuntimeError,))

    def _boom(*a, **k):
        raise ValueError("unrelated")

    monkeypatch.setattr(m, "run_analysis", _boom)
    result = CliRunner().invoke(m.app, [])
    assert isinstance(result.exception, ValueError)
