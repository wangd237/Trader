"""供全套 pytest 共享的 Fixture，防止缺少 API Key 时 CI 卡住。"""

import os
import socket

import pytest


def _blank_settings_overlay():
    """在导入包前清空所有 TRADINGAGENTS_* 环境变量。

    导入包时会加载 .env，并将这些变量并入 DEFAULT_CONFIG。
    如果不隔离，开发者自己的设置会影响测试默认值。
    将变量设置为空字符串可以防止 load_dotenv 再次覆盖，
    覆盖逻辑也会将空字符串视为未设置。
    专门验证环境覆盖的测试会自行设置需要的值。"""
    from dotenv import dotenv_values, find_dotenv

    names = set(os.environ)
    for filename in (".env", ".env.enterprise"):
        names |= set(dotenv_values(find_dotenv(filename, usecwd=True)))
    for name in names:
        if name.startswith("TRADINGAGENTS_"):
            os.environ[name] = ""


_blank_settings_overlay()


def _own_file_locations():
    """将测试结果、缓存和 Memory Log 限制在专用目录中。

    必须在导入包之前完成，否则默认路径可能指向用户主目录，
    读取到用户真实运行时生成的缓存。"""
    import tempfile

    home = tempfile.mkdtemp(prefix="tradingagents-tests-")
    os.environ["TRADINGAGENTS_RESULTS_DIR"] = os.path.join(home, "logs")
    os.environ["TRADINGAGENTS_CACHE_DIR"] = os.path.join(home, "cache")
    os.environ["TRADINGAGENTS_MEMORY_LOG_PATH"] = os.path.join(home, "memory", "trading_memory.md")


_own_file_locations()


def pytest_configure(config):
    for marker in ("unit", "integration", "smoke"):
        config.addinivalue_line("markers", f"{marker}: {marker}-level tests")
    # yfinance 默认把各标的时区信息缓存到用户主目录；
    # 测试使用独立目录隔离这类缓存。
    import tempfile

    import yfinance

    yfinance.set_tz_cache_location(tempfile.mkdtemp(prefix="tradingagents-tests-yf-"))


@pytest.fixture(autouse=True)
def _no_network(request, monkeypatch):
    """测试默认禁止联网；确实需要网络的测试必须标记 integration。"""
    if request.node.get_closest_marker("integration"):
        return

    def refuse(*args, **kwargs):
        raise OSError(f"test tried to reach the network: {args[1:] or kwargs}")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: refuse(None, *a))
    # yfinance 的 HTTP 客户端使用 libcurl，不依赖 Python 的 socket。
    from curl_cffi import requests as curl_requests

    monkeypatch.setattr(curl_requests.Session, "request", refuse)
    monkeypatch.setattr(curl_requests.AsyncSession, "request", refuse)


@pytest.fixture(autouse=True)
def _at_a_terminal(monkeypatch):
    """测试交互步骤时模拟真实终端，因为 pytest 的 stdin 不是终端。
    测试无交互运行的用例会主动将 isatty 设置为 False。"""
    import sys

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)


@pytest.fixture(autouse=True)
def _own_cli_prefs(tmp_path, monkeypatch):
    """CLI 将上次选择保存在主目录，测试必须与用户真实配置隔离。"""
    monkeypatch.setattr("cli.prefs._PREFS_PATH", tmp_path / "cli_prefs.json")


_API_KEY_ENV_VARS = (
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "ANTHROPIC_API_KEY",
    "XAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "DASHSCOPE_API_KEY",
    "DASHSCOPE_CN_API_KEY",
    "ZHIPU_API_KEY",
    "ZHIPU_CN_API_KEY",
    "MINIMAX_API_KEY",
    "MINIMAX_CN_API_KEY",
    "OPENROUTER_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "ALPHA_VANTAGE_API_KEY",
)


@pytest.fixture(autouse=True)
def _dummy_api_keys(monkeypatch):
    for env_var in _API_KEY_ENV_VARS:
        # 使用 `or` 而非 .get 的默认参数：
        # 如果环境变量存在但为空（例如从 env.example 复制的空密钥），
        # 仍需使用测试用的占位值。
        monkeypatch.setenv(env_var, os.environ.get(env_var) or "placeholder")


@pytest.fixture(autouse=True)
def _isolate_config():
    """每个测试执行前后重置全局 dataflows 配置。

    ``set_config`` 只合并键，不清除本次覆盖中不存在的配置。
    某个测试设置 tool_vendors 等字段后，可能污染后续测试，
    让路由结果受测试执行顺序影响。
    因此直接恢复干净的 DEFAULT_CONFIG，而不是局部合并。"""
    import copy

    import tradingagents.dataflows.config as config_module
    import tradingagents.default_config as default_config

    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)
    yield
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)
