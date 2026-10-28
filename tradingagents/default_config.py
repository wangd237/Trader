import os

from tradingagents.unified_llm import apply_unified_config

_TRADINGAGENTS_HOME = os.path.join(os.path.expanduser("~"), ".tradingagents")

# 环境变量与配置键的统一映射表。要支持新的环境变量覆盖项，
# 只需在这里添加一行，无须修改各入口代码。
# 类型转换依据已有默认值，因此 .env 中仍可写普通字符串。
_ENV_OVERRIDES = {
    "TRADINGAGENTS_LLM_PROVIDER":         "llm_provider",
    "TRADINGAGENTS_DEEP_THINK_LLM":       "deep_think_llm",
    "TRADINGAGENTS_QUICK_THINK_LLM":      "quick_think_llm",
    "TRADINGAGENTS_DEEP_THINK_PROVIDER":       "deep_think_provider",
    "TRADINGAGENTS_QUICK_THINK_PROVIDER":      "quick_think_provider",
    "TRADINGAGENTS_DEEP_THINK_BACKEND_URL":    "deep_think_backend_url",
    "TRADINGAGENTS_QUICK_THINK_BACKEND_URL":   "quick_think_backend_url",
    "TRADINGAGENTS_LLM_BACKEND_URL":      "backend_url",
    "TRADINGAGENTS_OUTPUT_LANGUAGE":      "output_language",
    "TRADINGAGENTS_MAX_DEBATE_ROUNDS":    "max_debate_rounds",
    "TRADINGAGENTS_MAX_RISK_ROUNDS":      "max_risk_discuss_rounds",
    "TRADINGAGENTS_MAX_TOOL_ROUNDS":      "max_tool_rounds",
    "TRADINGAGENTS_CHECKPOINT_ENABLED":   "checkpoint_enabled",
    "TRADINGAGENTS_BENCHMARK_TICKER":     "benchmark_ticker",
    "TRADINGAGENTS_TEMPERATURE":          "temperature",
    "TRADINGAGENTS_LLM_MAX_RETRIES":      "llm_max_retries",
    "TRADINGAGENTS_MAX_TOKENS":           "max_tokens",
    # 提供商专属推理参数，None 表示使用提供商默认值。
    # 这里可供非交互式运行配置；CLI 也提供交互式选择，
    # 设置对应环境变量后会跳过相应提问。
    "TRADINGAGENTS_GOOGLE_THINKING_LEVEL":   "google_thinking_level",
    "TRADINGAGENTS_OPENAI_REASONING_EFFORT": "openai_reasoning_effort",
    "TRADINGAGENTS_ANTHROPIC_EFFORT":        "anthropic_effort",
}


_BOOL_TRUE = ("true", "1", "yes", "on")
_BOOL_FALSE = ("false", "0", "no", "off")


def _coerce(value: str, reference):
    """将环境变量字符串转换成现有默认值的数据类型。

    非法输入会直接抛出 ``ValueError``，而不会静默使用默认值；
    例如布尔值误拼为 ``treu`` 或整数含有非数字字符，应在启动时明确报错，
    避免无人值守任务使用错误配置。"""
    if isinstance(reference, bool):
        normalized = value.strip().lower()
        if normalized in _BOOL_TRUE:
            return True
        if normalized in _BOOL_FALSE:
            return False
        raise ValueError(
            f"expected a boolean ({'/'.join(_BOOL_TRUE + _BOOL_FALSE)}), got {value!r}"
        )
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    return value


def _apply_env_overrides(config: dict) -> dict:
    """将 TRADINGAGENTS_* 环境变量覆盖到配置字典中（原地修改）。"""
    for env_var, key in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_var)
        if raw is None or raw == "":
            continue
        try:
            config[key] = _coerce(raw, config.get(key))
        except ValueError as exc:
            raise ValueError(f"Invalid value for {env_var}: {exc}") from exc
    return config


def build_default_config() -> dict:
    """生成内置默认配置并应用 TRADINGAGENTS_* 环境变量覆盖。

    模块导入时调用并赋值给 DEFAULT_CONFIG；需要读取当前环境变量时
    可以再次调用此函数。"""
    return apply_unified_config(_apply_env_overrides({
        "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR") or os.path.join(_TRADINGAGENTS_HOME, "logs"),
        "data_cache_dir": os.getenv("TRADINGAGENTS_CACHE_DIR") or os.path.join(_TRADINGAGENTS_HOME, "cache"),
        "memory_log_path": os.getenv("TRADINGAGENTS_MEMORY_LOG_PATH") or os.path.join(_TRADINGAGENTS_HOME, "memory", "trading_memory.md"),
        # 可选的已结算记忆条目数量上限；超过上限后清理最早的已结算记录。
        # 待结算记录永不清理。None 表示不限制数量。
        "memory_log_max_entries": None,
        # LLM 配置。
        "llm_provider": "openai",
        "deep_think_llm": "gpt-6-sol",
        "quick_think_llm": "gpt-6-luna",
        # 设置 None 时，每个提供商使用自己的默认端点，例如 OpenAI 的
        # api.openai.com 和 Gemini 的 generativelanguage.googleapis.com。
        # CLI 可按用户选择覆盖。这里不能固定特定提供商的 URL，
        # 否则例如会将 OpenAI 的 /v1 路径错误传给 Gemini，造成非法请求地址。
        "backend_url": None,
        # quick 和 deep 层级可以指定独立提供商及端点（#1440）。
        # quick 服务于分析师、研究员、辩论员和 Trader；deep 服务于管理者。
        # None 表示沿用 llm_provider 与 backend_url。
        "quick_think_provider": None,
        "deep_think_provider": None,
        "quick_think_backend_url": None,
        "deep_think_backend_url": None,
        # 提供商专属推理配置。
        "google_thinking_level": None,      # "high"、"minimal" 等
        "openai_reasoning_effort": None,    # "medium", "high", "low"
        "anthropic_effort": None,           # "high", "medium", "low"
        # temperature 设置后透传给所有提供商，None 保留各自默认值。
        # 对支持它的模型，低温度通常能降低多次运行的输出差异；
        # 推理模型可能忽略该参数，且无法保证输出完全一致（见 README）。
        "temperature": None,
        # SDK 重试预算作用于所有 LLM 客户端，None 表示使用提供商或 SDK
        # 默认值（通常为 2）。限流较频繁的部署可增大该值，
        # 避免短暂的 429 错误导致运行中止（#1091）。
        "llm_max_retries": None,
        # 输出 token 上限传给所有 LLM 客户端；None 使用各自默认值。
        # 可限制模型过长的推理或输出，避免调用挂起或触发网关空闲超时
        # （例如部分 deepseek-v4-flash 部署，#1204）。
        "max_tokens": None,
        # 检查点与恢复：True 表示 LangGraph 在每个节点结束后保存状态，
        # 崩溃后可从上次成功的节点继续。
        "checkpoint_enabled": False,
        # 分析师报告及最终决策的输出语言。
        # Agent 内部辩论保留英文，以维持推理效果。
        "output_language": "English",
        # 辩论与讨论参数。
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
        "max_recur_limit": 100,
        # 分析师被要求提交报告前允许的最大工具调用轮数。
        "max_tool_rounds": 20,
        # 新闻和数据获取参数。
        # 增大参数可支持更长时间回溯或更广泛的宏观覆盖，
        # 减小参数可降低 Agent Prompt 的 token 开销。
        "news_article_limit": 20,             # 每个 ticker 的最大新闻数
        "global_news_article_limit": 10,      # 全球或宏观新闻的最大文章数
        "global_news_lookback_days": 7,       # 宏观新闻回溯天数
        # get_global_news 检索宏观新闻时使用的关键词。
        # 可增加或替换，以扩大地区、行业覆盖范围。
        "global_news_queries": [
            "Federal Reserve interest rates inflation",
            "S&P 500 earnings GDP economic outlook",
            "geopolitical risk trade war sanctions",
            "ECB Bank of England BOJ central bank policy",
            "oil commodities supply chain energy",
        ],
        # 数据提供商配置。
        # 类别级配置为该类别全部工具提供默认数据源。
        # 提供商列表即准确的回退顺序，系统不会自动使用用户未选择的提供商；
        # 例如 "yfinance,alpha_vantage" 表示按顺序尝试，
        # "default" 表示采用全部可用提供商。
        "data_vendors": {
            "core_stock_apis": "yfinance",       # 可选：alpha_vantage、yfinance
            "technical_indicators": "yfinance",  # 可选：alpha_vantage、yfinance
            # 美国上市公司财务报表优先使用 SEC EDGAR 的原始披露文件，
            # 之后回退到 Yahoo。公司概况及内部人交易等 SEC 不覆盖的工具
            # 则使用 Yahoo。
        "fundamental_data": "sec_edgar,yfinance",  # 可选：sec_edgar、alpha_vantage、yfinance
            "news_data": "yfinance",             # 可选：alpha_vantage、yfinance
            "macro_data": "fred",                # 可选：fred（需要 FRED_API_KEY）
            "prediction_markets": "polymarket",  # 可选：polymarket（无需密钥）
        },
        # 工具级配置（优先于类别级配置）。
        "tool_vendors": {
            # A 股可通过 tool_vendors 单项设置为 ashare；海外默认配置保持不变。
            # 示例："get_stock_data": "alpha_vantage",  # 覆盖类别默认提供商
        },
        # 反思层计算 Alpha 时使用的基准。
        # 设置 ``benchmark_ticker`` 时，所有 ticker 使用同一个基准；
        # 留空时由 ``benchmark_map`` 根据交易所后缀自动选择指数。
        # 美股默认使用 SPY，因此反思会显示 "Alpha vs SPY"；
        # 其他市场自动选用其地区指数。
        #
        # ``holding_period_days`` 是从分析日期之后的第一个收盘日算起的交易日数，
        # 用来衡量决策表现，并用于反思和回测统计。
        "holding_period_days": 5,
        "benchmark_ticker": None,
        "benchmark_map": {
            ".NS":  "^NSEI",       # 印度 NSE（Nifty 50）
            ".BO":  "^BSESN",      # 印度 BSE（Sensex）
            ".T":   "^N225",       # 东京（日经 225）
            ".TW":  "^TWII",       # 台湾（TAIEX）
            ".TWO": "^TWII",       # 台北柜买市场（Yahoo 无 TPEx 指数，采用 TAIEX）
            ".KS":  "^KS11",       # 韩国（KOSPI）
            ".KQ":  "^KQ11",       # 韩国（KOSDAQ）
            ".HK":  "^HSI",        # 香港（恒生指数）
            ".SI":  "^STI",        # 新加坡（海峡时报指数）
            ".L":   "^FTSE",       # 伦敦（FTSE 100）
            ".DE":  "^GDAXI",      # 德国（DAX）
            ".PA":  "^FCHI",       # 巴黎（CAC 40）
            ".AS":  "^AEX",        # 阿姆斯特丹（AEX）
            ".SW":  "^SSMI",       # 瑞士（SMI）
            ".MI":  "FTSEMIB.MI",  # 米兰（FTSE MIB）
            ".TO":  "^GSPTSE",     # 多伦多（TSX 综合指数）
            ".AX":  "^AXJO",       # 澳大利亚（ASX 200）
            ".SS":  "000001.SS",   # 上海（上证综指）
            ".SZ":  "399001.SZ",   # 深圳（深证成指）
            ".SA":  "^BVSP",       # 巴西 B3（Ibovespa）
            "":     "SPY",         # 美股默认基准（无后缀）
        },

    }))


DEFAULT_CONFIG = build_default_config()
