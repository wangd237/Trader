"""测试套件必须在不同机器上保持一致，不依赖网络或用户真实文件。"""

import socket

import pytest


@pytest.mark.unit
@pytest.mark.parametrize("connect", [
    lambda: socket.create_connection(("192.0.2.1", 80), timeout=1),
    lambda: socket.socket().connect_ex(("192.0.2.1", 80)),
    lambda: socket.getaddrinfo("example.com", 443),
    # yfinance 使用 libcurl 发送 HTTP 请求，不经过 Python socket。
    lambda: __import__("curl_cffi.requests").requests.Session().get("http://192.0.2.1", timeout=1),
], ids=["connect", "connect_ex", "dns", "curl_cffi"])
def test_a_test_cannot_reach_the_network(connect):
    """意外依赖真实数据源的测试可能随网络环境忽好忽坏；
    因此 conftest 阻止此类联网。"""
    with pytest.raises(OSError, match="reach the network"):
        connect()


@pytest.mark.unit
def test_a_test_saves_cli_selections_to_its_own_directory(tmp_path):
    """CLI 会把上次选择保存到用户主目录；测试选择流程不能覆盖真实偏好文件。"""
    from cli import prefs

    prefs.save_last_run({"analysts": ["market"]})

    assert prefs._PREFS_PATH.is_relative_to(tmp_path)
    assert prefs._PREFS_PATH.exists()


@pytest.mark.unit
def test_yfinance_keeps_its_cache_out_of_the_users_home():
    """yfinance 默认将标的时区缓存到用户主目录。"""
    from pathlib import Path

    from yfinance.cache import _TzDBManager

    assert not Path(_TzDBManager.get_location()).is_relative_to(Path.home())


@pytest.mark.unit
@pytest.mark.parametrize("key", ["results_dir", "data_cache_dir", "memory_log_path"])
def test_the_suite_keeps_its_files_out_of_the_users_home(key):
    """读取用户缓存会使测试结果取决于历史运行数据；
    写入该目录甚至可能修改用户真实缓存。"""
    from pathlib import Path

    from tradingagents.default_config import DEFAULT_CONFIG

    assert not Path(DEFAULT_CONFIG[key]).is_relative_to(Path.home() / ".tradingagents")
