from contextlib import contextmanager
from contextvars import Context, ContextVar, copy_context
from copy import deepcopy

import tradingagents.default_config as default_config

# 使用默认配置，同时允许后续覆盖。
_config: dict | None = None

# 这是当前运行独立的配置。Graph 在执行期间绑定自己的配置，
# 因此即使多个 Graph 共用一个进程，各自调用的数据工具仍读取
# 对应 Graph 的数据源设置。LangGraph 会将上下文传递给工具调用。
_run_config: ContextVar[dict | None] = ContextVar("tradingagents_run_config", default=None)


def initialize_config():
    """使用默认值初始化配置。"""
    global _config
    if _config is None:
        _config = deepcopy(default_config.DEFAULT_CONFIG)


def _merge(base: dict, config: dict) -> dict:
    """将 ``config`` 合并到 ``base``：字典字段只做一层嵌套合并，标量直接替换。"""
    for key, value in deepcopy(config).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key].update(value)
        else:
            base[key] = value
    return base


def set_config(config: dict):
    """使用自定义值更新配置。
    
    字典类型字段（如 ``data_vendors``）只合并一层，因此局部更新
    ``{"data_vendors": {"core_stock_apis": "alpha_vantage"}}``
    会保留默认配置中的其他数据类别；标量字段则直接覆盖。"""
    initialize_config()
    _merge(_config, config)


@contextmanager
def run_config(config: dict):
    """在上下文块中，每次读取配置都使用 ``config`` 覆盖默认值。"""
    token = _run_config.set(_merge(deepcopy(default_config.DEFAULT_CONFIG), config))
    try:
        yield
    finally:
        _run_config.reset(token)


def run_config_context(config: dict) -> Context:
    """在上下文内的所有配置读取都使用 ``config`` 覆盖默认配置。
    
    适用于步骤之间会让出控制权的执行过程：普通 ``run_config`` 块
    可能在调用者持有结果时仍保持开启，不便表达这种执行边界。"""
    context = copy_context()
    context.run(_run_config.set, _merge(deepcopy(default_config.DEFAULT_CONFIG), config))
    return context


def get_config() -> dict:
    """优先返回当前运行的配置，否则返回进程全局配置。"""
    scoped = _run_config.get()
    if scoped is not None:
        return deepcopy(scoped)
    if _config is None:
        initialize_config()
    return deepcopy(_config)


initialize_config()
