import datetime
import os
import subprocess
import sys
from pathlib import Path

import questionary
import typer
from dotenv import find_dotenv, set_key
from rich.markup import escape

from cli.display import console
from cli.models import AnalystType, AssetType
from tradingagents.llm_clients.api_key_env import get_api_key_env
from tradingagents.llm_clients.model_catalog import get_model_options
from tradingagents.llm_clients.openai_client import OPENAI_COMPATIBLE_PROVIDERS, SIGNED_IN_PROVIDERS

TICKER_INPUT_EXAMPLES = "SPY, 0700.HK, BTC-USD"

ANALYST_CHOICES = [
    ("Market Analyst", AnalystType.MARKET),
    ("Sentiment Analyst", AnalystType.SOCIAL),
    ("News Analyst", AnalystType.NEWS),
    ("Fundamentals Analyst", AnalystType.FUNDAMENTALS),
]

CRYPTO_SUFFIXES = ("-USD", "-USDT", "-USDC", "-BTC", "-ETH")


# 统一各个选项菜单的显示样式。
_MENU_STYLE = questionary.Style([
    ("selected", "fg:magenta noinherit"),
    ("highlighted", "fg:magenta noinherit"),
    ("pointer", "fg:magenta noinherit"),
])


def is_valid_ticker_input(value: str) -> bool:
    """检查 ticker 输入是否满足允许字符和长度要求。

    允许 Yahoo 使用的特殊字符，例如期货或外汇代码中的 ``=``
    （如 ``GC=F``、``EURUSD=X``，#980），以及指数代码中的 ``^``。
    允许空输入，后续将默认使用 SPY。"""
    v = value.strip()
    return not v or (all(ch.isalnum() or ch in "._-^=" for ch in v) and len(v) <= 32)


def get_ticker() -> str:
    """提示用户输入 ticker，并保留交易所后缀。

    使用 questionary.text 而非 typer.prompt：后者在某些 Shell 下
    可能错误处理 ``000404.SH`` 这样的末尾点后缀。
    同时检查允许字符，提前发现明显的输入错误。"""
    ticker = questionary.text(
        f"Enter ticker symbol (e.g. {TICKER_INPUT_EXAMPLES}):",
        validate=lambda x: (
            is_valid_ticker_input(x)
            or "Please enter a valid ticker symbol, e.g. AAPL, 000404.SZ, 0700.HK, GC=F."
        ),
        style=questionary.Style(
            [
                ("text", "fg:green"),
                ("highlighted", "noinherit"),
            ]
        ),
    ).ask()

    if ticker is None:
        console.print("\n[red]No ticker symbol provided. Exiting...[/red]")
        exit(1)

    return normalize_ticker_symbol(ticker) if ticker.strip() else "SPY"


def parse_ticker(value: str) -> str:
    """规范化命令行传入的 ticker，拒绝空值或格式错误的值。"""
    if not value.strip() or not is_valid_ticker_input(value):
        raise ValueError(f"not a ticker symbol: {value!r} (e.g. {TICKER_INPUT_EXAMPLES})")
    return normalize_ticker_symbol(value)


def parse_analysis_date(value: str) -> str:
    """校验分析日期格式 YYYY-MM-DD，且不得晚于今天。"""
    try:
        day = datetime.datetime.strptime(value.strip(), "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"not a date: {value!r}; use YYYY-MM-DD") from None
    if day > datetime.date.today():
        raise ValueError(f"{value} is in the future")
    return day.isoformat()


def parse_analysts(value: str, asset_type: AssetType) -> list[AnalystType]:
    """解析逗号分隔的分析师名称，按固定顺序排列，并校验资产类型是否支持。"""
    # 用户看到的名称为 "sentiment"，分析师内部标识则为 "social"。
    names = [{"sentiment": "social"}.get(n, n) for n in (n.strip().lower() for n in value.split(",")) if n]
    if not names:
        raise ValueError("name at least one analyst")
    known = {a.value: a for a in AnalystType}
    unknown = [n for n in names if n not in known]
    if unknown:
        choices = ", ".join("sentiment" if name == "social" else name for name in known)
        raise ValueError(f"unknown analyst {', '.join(unknown)}; choose from {choices}")
    available = filter_analysts_for_asset_type(list(known.values()), asset_type)
    unavailable = [n for n in names if known[n] not in available]
    if unavailable:
        raise ValueError(f"{', '.join(unavailable)} is not available for {asset_type.value}")
    return [a for a in available if a.value in names]


def normalize_ticker_symbol(ticker: str) -> str:
    """将用户输入转换成标准 Yahoo 代码，统一符号映射来源。

    调用数据层的 ``normalize_symbol``，确保 CLI 传入的标的代码
    与行情数据查询完全一致，例如 ``BTCUSD`` → ``BTC-USD``，
    ``XAUUSD`` → ``GC=F``。若数据层不可用，退回简单的大写转换。"""
    try:
        from tradingagents.dataflows.symbols import normalize_symbol

        return normalize_symbol(ticker)
    except Exception:
        return ticker.strip().upper()


def detect_asset_type(ticker: str) -> AssetType:
    """依据标准化代码判断资产类型，使 BTCUSD 和 BTC-USDT
    都正确识别为加密资产（#981/#982），与实际数据查询一致。"""
    canonical = normalize_ticker_symbol(ticker)
    if canonical.endswith(CRYPTO_SUFFIXES):
        return AssetType.CRYPTO
    return AssetType.STOCK


def filter_analysts_for_asset_type(
    analysts: list[AnalystType], asset_type: AssetType
) -> list[AnalystType]:
    if asset_type != AssetType.CRYPTO:
        return analysts
    return [
        analyst
        for analyst in analysts
        if analyst != AnalystType.FUNDAMENTALS
    ]


def _matching_choice(options, default):
    """返回与 ``default`` 相匹配的选项值；若无匹配则返回 None，不更改菜单。"""
    return next((value for _, value in options if value == default), None)


def select_analysts(asset_type: AssetType = AssetType.STOCK, default=None) -> list[AnalystType]:
    """通过交互式复选框选择分析师。

    ``default`` 仅用于预选上次运行的分析师，提示仍然显示。"""
    available_analysts = filter_analysts_for_asset_type(
        [value for _, value in ANALYST_CHOICES],
        asset_type,
    )
    choices = questionary.checkbox(
        "Select Your [Analysts Team]:",
        choices=[
            questionary.Choice(display, value=value, checked=value.value in (default or []))
            for display, value in ANALYST_CHOICES
            if value in available_analysts
        ],
        instruction="\n- Press Space to select/unselect analysts\n- Press 'a' to select/unselect all\n- Press Enter when done",
        validate=lambda x: len(x) > 0 or "You must select at least one analyst.",
        style=questionary.Style(
            [
                ("checkbox-selected", "fg:green"),
                ("selected", "fg:green noinherit"),
                ("highlighted", "noinherit"),
                ("pointer", "noinherit"),
            ]
        ),
    ).ask()

    if not choices:
        console.print("\n[red]No analysts selected. Exiting...[/red]")
        exit(1)

    return choices


def select_research_depth(default=None) -> int:
    """通过交互式菜单选择研究深度。"""

    DEPTH_OPTIONS = [
        ("Shallow - Quick research, few debate and strategy discussion rounds", 1),
        ("Medium - Middle ground, moderate debate rounds and strategy discussion", 3),
        ("Deep - Comprehensive research, in depth debate and strategy discussion", 5),
    ]

    choice = questionary.select(
        "Select Your [Research Depth]:",
        choices=[
            questionary.Choice(display, value=value) for display, value in DEPTH_OPTIONS
        ],
        default=_matching_choice(DEPTH_OPTIONS, default),
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=questionary.Style(
            [
                ("selected", "fg:yellow noinherit"),
                ("highlighted", "fg:yellow noinherit"),
                ("pointer", "fg:yellow noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print("\n[red]No research depth selected. Exiting...[/red]")
        exit(1)

    return choice


# 主流 OpenRouter 聊天模型提供商的命名空间。优先展示这些提供商
# 的最新通用模型，而不是整个市场的最新模型，后者容易被实验性、
# 小众或专用版本占据。其他提供商仍可通过自定义 ID 使用。
# 提供商名称相对稳定，维护成本低于不断变化的模型 ID。
_OPENROUTER_MAINSTREAM = {
    "openai", "anthropic", "google", "deepseek", "qwen", "mistralai",
    "meta-llama", "x-ai", "z-ai", "minimax", "moonshotai",
}


def _fetch_openrouter_models() -> list[tuple[str, str]]:
    """从 OpenRouter API 获取可用模型列表。"""
    import requests
    try:
        resp = requests.get("https://openrouter.ai/api/v1/models", timeout=10)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        # 按照时间倒序排序，确保 top-N 真的是最新模型。
        # 即使当前 API 默认按时间返回，也需显式排序以保证菜单承诺。
        models.sort(key=lambda m: m.get("created") or 0, reverse=True)
        return [(m.get("name") or m["id"], m["id"]) for m in models]
    except Exception as e:
        console.print(f"\n[yellow]Could not fetch OpenRouter models: {e}[/yellow]")
        return []


def select_chatgpt_model(mode: str, default=None) -> str:
    """选择 ChatGPT 订阅可用的模型，或手动输入模型 ID。"""
    from tradingagents.llm_clients import chatgpt_auth
    try:
        models = chatgpt_auth.list_models()
    except Exception as e:
        console.print(f"\n[yellow]Could not list your ChatGPT plan's models: {escape(str(e))}[/yellow]")
        models = []
    return _select_listed_model(f"Select Your [{mode.title()}-Thinking] ChatGPT Plan Model:",
                                models, default, _prompt_custom_model_id)


def _select_listed_model(title: str, models, default, custom) -> str:
    """从实时获取的 ``models``（名称、模型 ID 配对）中选择，
    或通过 ``custom`` 输入自定义模型 ID。"""
    choices = [questionary.Choice(name, value=mid) for name, mid in models]
    choices.append(questionary.Choice("Custom model ID", value="custom"))
    choice = questionary.select(
        title,
        choices=choices,
        default=default if any(c.value == default for c in choices) else None,
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=_MENU_STYLE,
    ).ask()
    if choice is None:
        console.print("\n[red]No model selected. Exiting...[/red]")
        exit(1)
    return custom() if choice == "custom" else choice


def _require_text(message: str, hint: str) -> str:
    """提示输入必填项；用户取消时安全退出。

    ``questionary.text(...).ask()`` 在 Ctrl-C/Esc 时返回 None。
    与其他必填项保持一致，取消时直接退出，避免空模型或空部署名称
    继续传入下游导致错误。"""
    response = questionary.text(
        message,
        validate=lambda x: len(x.strip()) > 0 or hint,
    ).ask()
    if response is None:
        console.print("\n[red]Cancelled. Exiting...[/red]")
        exit(1)
    return response.strip()


def select_openrouter_model(mode: str, default=None) -> str:
    """从 OpenRouter 最新可用模型中选择，或输入自定义 ID。

    ``mode``（"quick"/"deep"）用于区分连续出现的两次模型选择，
    与其他提供商的交互逻辑一致（#1000）。"""
    models = _fetch_openrouter_models()  # newest first
    # 优先选取主流提供商最新模型，避免候选列表被小众或实验版本占满；
    # 如果没有匹配项，再回退到全部模型。
    mainstream = [
        (name, mid) for name, mid in models
        if not mid.startswith("~")  # skip variant/alias duplicate routes
        and mid.split("/", 1)[0] in _OPENROUTER_MAINSTREAM
    ]
    top = (mainstream or models)[:5]
    return _select_listed_model(
        f"Select Your [{mode.title()}-Thinking] OpenRouter Model (latest available):", top, default,
        lambda: _require_text("Enter OpenRouter model ID (e.g. google/gemma-4-26b-a4b-it):",
                              "Please enter a model ID."))


# 支持实时获取模型列表的提供商，以及相应的选择菜单。
LISTED_LIVE = {"openrouter": select_openrouter_model, "chatgpt": select_chatgpt_model}


def _prompt_custom_model_id() -> str:
    """提示用户输入自定义模型 ID。"""
    return _require_text("Enter model ID:", "Please enter a model ID.")


def _select_model(provider: str, mode: str, default=None) -> str:
    """为指定提供商和 quick/deep 模式选择模型。"""
    listed = LISTED_LIVE.get(provider.lower())
    if listed is not None:
        return listed(mode, default)

    if provider.lower() == "azure":
        return _require_text(
            f"Enter Azure deployment name ({mode}-thinking):",
            "Please enter a deployment name.",
        )

    choice = questionary.select(
        f"Select Your [{mode.title()}-Thinking LLM Engine]:",
        choices=[
            questionary.Choice(display, value=value)
            for display, value in get_model_options(provider, mode)
        ],
        default=_matching_choice(get_model_options(provider, mode), default),
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=_MENU_STYLE,
    ).ask()

    if choice is None:
        console.print(f"\n[red]No {mode} thinking llm engine selected. Exiting...[/red]")
        exit(1)

    if choice == "custom":
        return _prompt_custom_model_id()

    return choice


def select_shallow_thinking_agent(provider, default=None) -> str:
    """通过交互式菜单选择轻量推理模型。"""
    return _select_model(provider, "quick", default)


def select_deep_thinking_agent(provider, default=None) -> str:
    """通过交互式菜单选择深度推理模型。"""
    return _select_model(provider, "deep", default)


def _llm_provider_table() -> list[tuple[str, str, str | None]]:
    """列出所有提供商的 (display_name, provider_key, base_url)。

    交互式选择和环境变量驱动的配置共用这份列表，确保使用环境变量
    指定提供商时，也能解析到与菜单相同的默认端点。
    Ollama 可通过 OLLAMA_BASE_URL 连接远程 ollama-serve，
    未设置时使用 localhost 默认地址。"""
    ollama_url = os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434/v1"
    return [
        ("OpenAI", "openai", "https://api.openai.com/v1"),
        ("Google", "google", None),
        ("Anthropic", "anthropic", "https://api.anthropic.com/"),
        ("ChatGPT (Plus or Pro plan)", "chatgpt", "https://api.openai.com/v1"),
        ("xAI", "xai", "https://api.x.ai/v1"),
        ("DeepSeek", "deepseek", "https://api.deepseek.com"),
        ("Qwen", "qwen", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
        # ZHIPU_API_KEY 对应 Z.AI 国际平台；中国地区单独使用 glm-cn 标识，
        # 通过后续地区选择菜单指定。
        ("GLM", "glm", "https://api.z.ai/api/paas/v4/"),
        ("MiniMax", "minimax", "https://api.minimax.io/v1"),
        ("OpenRouter", "openrouter", "https://openrouter.ai/api/v1"),
        ("Mistral", "mistral", "https://api.mistral.ai/v1"),
        ("Kimi (Moonshot)", "kimi", "https://api.moonshot.ai/v1"),
        ("Groq", "groq", "https://api.groq.com/openai/v1"),
        ("NVIDIA NIM", "nvidia", "https://integrate.api.nvidia.com/v1"),
        ("Azure OpenAI", "azure", None),
        ("Amazon Bedrock", "bedrock", None),
        ("Ollama", "ollama", ollama_url),
        ("OpenAI-compatible (vLLM, LM Studio, llama.cpp, custom relay)", "openai_compatible", None),
    ]


def provider_default_url(provider_key: str) -> str | None:
    """返回指定提供商的默认后端 URL；未知提供商返回 None。"""
    key = provider_key.lower()
    for _, pk, url in _llm_provider_table():
        if pk == key:
            return url
    return None


def resolve_backend_url(
    provider: str, menu_url: str | None = None, env_url: str | None = None
) -> str | None:
    """按正确优先级解析后端 URL。

    显式环境变量覆盖（``env_url``，来自
    ``TRADINGAGENTS_LLM_BACKEND_URL`` 经 DEFAULT_CONFIG 传递）
    无论提供商是交互选择还是由环境变量设置都始终优先（#978）。
    否则依次使用菜单/地区端点和提供商默认端点。"""
    return env_url or menu_url or provider_default_url(provider)


def prompt_openai_compatible_url(default=None) -> str:
    """提示输入自定义 OpenAI 兼容接口的基础 URL。"""
    url = questionary.text(
        "Enter the OpenAI-compatible base URL "
        "(e.g. http://localhost:8000/v1 for vLLM, http://localhost:1234/v1 for LM Studio):",
        default=default or "",
        validate=lambda x: x.strip().startswith(("http://", "https://"))
        or "Enter a URL starting with http:// or https://",
    ).ask()
    if not url:
        console.print("\n[red]No endpoint URL provided. Exiting...[/red]")
        exit(1)
    return url.strip()


# 把通过个人订阅计划使用的提供商归为同一个菜单选项。
_SUBSCRIPTION = ("subscription", None)


def select_llm_provider(default=None) -> tuple[str, str | None]:
    """选择 LLM 提供商及其 API 端点。"""
    PROVIDERS = _llm_provider_table()
    # qwen-cn 等地区专属标识在下一步选择，主菜单仅展示基础提供商。
    base = (default or "").split("-cn")[0]
    choices = []
    for display, provider_key, url in PROVIDERS:
        if provider_key in SIGNED_IN_PROVIDERS:
            continue
        choices.append(questionary.Choice(display, value=(provider_key, url)))
        if provider_key == "anthropic":
            choices.append(questionary.Choice("Subscription (use your ChatGPT plan)", value=_SUBSCRIPTION))
    preselected = _SUBSCRIPTION if base in SIGNED_IN_PROVIDERS else next(
        ((key, url) for _, key, url in PROVIDERS if key == base), None
    )

    choice = questionary.select(
        "Select your LLM Provider:",
        choices=choices,
        default=preselected,
        instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
        style=_MENU_STYLE,
    ).ask()

    if choice is None:
        console.print("\n[red]No LLM provider selected. Exiting...[/red]")
        exit(1)

    if choice == _SUBSCRIPTION:
        choice = questionary.select(
            "Select your subscription:",
            choices=[questionary.Choice(display, value=(key, url))
                     for display, key, url in PROVIDERS if key in SIGNED_IN_PROVIDERS],
            style=_MENU_STYLE,
        ).ask()
        if choice is None:
            console.print("\n[red]No subscription selected. Exiting...[/red]")
            exit(1)

    provider, url = choice
    return provider, url


def ask_openai_reasoning_effort() -> str:
    """询问 OpenAI 模型的推理力度级别。"""
    choices = [
        questionary.Choice("Medium (Default)", "medium"),
        questionary.Choice("High (More thorough)", "high"),
        questionary.Choice("Low (Faster)", "low"),
    ]
    return questionary.select(
        "Select Reasoning Effort:",
        choices=choices,
        style=questionary.Style([
            ("selected", "fg:cyan noinherit"),
            ("highlighted", "fg:cyan noinherit"),
            ("pointer", "fg:cyan noinherit"),
        ]),
    ).ask()


def ask_anthropic_effort() -> str | None:
    """询问 Anthropic 模型的推理力度级别。

    控制 Claude 4.5 / 4.6 / 4.7 的 token 用量和回答详尽程度。
    虽然 API 还接受 "max"，交互菜单只提供通用的 low/medium/high。"""
    return questionary.select(
        "Select Effort Level:",
        choices=[
        questionary.Choice("High (recommended)", "high"),
        questionary.Choice("Medium (balanced)", "medium"),
        questionary.Choice("Low (faster, cheaper)", "low"),
        ],
        style=questionary.Style([
            ("selected", "fg:cyan noinherit"),
            ("highlighted", "fg:cyan noinherit"),
            ("pointer", "fg:cyan noinherit"),
        ]),
    ).ask()


def ask_gemini_thinking_config() -> str | None:
    """询问 Gemini 的思考模式配置。

    返回 thinking_level（"high" 或 "minimal"）。
    客户端根据模型系列映射到相应 API 参数。"""
    return questionary.select(
        "Select Thinking Mode:",
        choices=[
        questionary.Choice("Enable Thinking (recommended)", "high"),
        questionary.Choice("Minimal/Disable Thinking", "minimal"),
        ],
        style=questionary.Style([
            ("selected", "fg:green noinherit"),
            ("highlighted", "fg:green noinherit"),
            ("pointer", "fg:green noinherit"),
        ]),
    ).ask()


def ask_glm_region() -> tuple[str, str]:
    """询问使用 GLM 的哪个平台：Z.AI 国际版或 BigModel 中国版。

    两个品牌提供相同 GLM 模型，但账户彼此独立，密钥不能互换。
    返回 (provider_key, backend_url)。"""
    return questionary.select(
        "Select GLM platform:",
        choices=[
            questionary.Choice(
                "Z.AI — api.z.ai (international, uses ZHIPU_API_KEY)",
                value=("glm", "https://api.z.ai/api/paas/v4/"),
            ),
            questionary.Choice(
                "BigModel — open.bigmodel.cn (China, uses ZHIPU_CN_API_KEY)",
                value=("glm-cn", "https://open.bigmodel.cn/api/paas/v4/"),
            ),
        ],
        style=questionary.Style([
            ("selected", "fg:cyan noinherit"),
            ("highlighted", "fg:cyan noinherit"),
            ("pointer", "fg:cyan noinherit"),
        ]),
    ).ask()


def ask_qwen_region() -> tuple[str, str]:
    """询问 Qwen 使用国际端点还是中国端点。

    阿里云 DashScope 的两个地区端点使用独立账户，密钥不能跨区使用
    （修复 #758）。返回 (provider_key, backend_url)。"""
    return questionary.select(
        "Select Qwen region:",
        choices=[
            questionary.Choice(
                "International — dashscope-intl.aliyuncs.com (uses DASHSCOPE_API_KEY)",
                value=("qwen", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
            ),
            questionary.Choice(
                "China — dashscope.aliyuncs.com (uses DASHSCOPE_CN_API_KEY)",
                value=("qwen-cn", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            ),
        ],
        style=questionary.Style([
            ("selected", "fg:cyan noinherit"),
            ("highlighted", "fg:cyan noinherit"),
            ("pointer", "fg:cyan noinherit"),
        ]),
    ).ask()


def ask_minimax_region() -> tuple[str, str]:
    """询问 MiniMax 使用国际还是中国端点。

    两个端点使用不同账户，密钥不能互换。
    返回 (provider_key, backend_url)。"""
    return questionary.select(
        "Select MiniMax region:",
        choices=[
            questionary.Choice(
                "Global — api.minimax.io (uses MINIMAX_API_KEY)",
                value=("minimax", "https://api.minimax.io/v1"),
            ),
            questionary.Choice(
                "China — api.minimaxi.com (uses MINIMAX_CN_API_KEY)",
                value=("minimax-cn", "https://api.minimaxi.com/v1"),
            ),
        ],
        style=questionary.Style([
            ("selected", "fg:cyan noinherit"),
            ("highlighted", "fg:cyan noinherit"),
            ("pointer", "fg:cyan noinherit"),
        ]),
    ).ask()


def confirm_ollama_endpoint(url: str) -> None:
    """在选择提供商后展示实际解析出的 Ollama 端点。

    模型选择前让用户看到三个信息：即将请求的 URL、地址来源
    （``OLLAMA_BASE_URL`` 或默认值），以及协议或端口可能不正确时的提示。
    警告仅供参考，不阻止执行，因为用户可能有意使用反向代理等特殊路径。"""
    from_env = os.environ.get("OLLAMA_BASE_URL")
    origin = " (from OLLAMA_BASE_URL)" if from_env and from_env == url else ""
    console.print(f"[green]✓ Using Ollama at {url}{origin}[/green]")

    if not url.startswith(("http://", "https://")):
        console.print(
            f"[yellow]Note: {url!r} is missing a scheme. "
            f"Ollama-serve typically expects a URL like "
            f"http://<host>:11434/v1.[/yellow]"
        )
    elif ":11434" not in url and "://localhost" not in url and "://127.0.0.1" not in url:
        # 端口不符合 ollama-serve 默认值且主机不是本地地址时给出提示；
        # 远程用户可能通过 80 端口等反向代理访问，因此仅提示不拦截。
        console.print(
            f"[yellow]Note: {url!r} doesn't include port 11434. "
            f"Make sure your remote ollama-serve listens on the port "
            f"shown above.[/yellow]"
        )


def ensure_signed_in(provider: str) -> None:
    """运行前验证订阅型提供商的登录状态，并允许现场登录。

    登录在独立进程（``tradingagents login chatgpt``）中进行，
    确保粘贴登录地址的输入操作不会影响后续交互问题。"""
    if provider.lower() not in SIGNED_IN_PROVIDERS:
        return
    from tradingagents.llm_clients import chatgpt_auth  # the one signed-in provider

    try:
        chatgpt_auth.access_token()
    except chatgpt_auth.ChatGPTUnreachable as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=1) from None
    except chatgpt_auth.ChatGPTAuthError as exc:
        if not sys.stdin.isatty():
            console.print(f"[red]{escape(str(exc))}[/red]")
            raise typer.Exit(code=1) from None
        console.print(f"\n[yellow]{escape(str(exc))}[/yellow]")
        # 使用 -P 防止工作目录中的同名 cli/ 包遮蔽项目自己的模块。
        login = [sys.executable, "-P", "-m", "cli.main", "login", "chatgpt"]
        if subprocess.run(login).returncode != 0:
            raise typer.Exit(code=1) from None
    console.print(f"[green]✓ Using your ChatGPT plan:[/green] {escape(chatgpt_auth.signed_in_as() or '')}")


def ensure_api_key(provider: str) -> str | None:
    """确保指定 `provider` 的 API Key 已存在于环境变量。

    若已设置则直接返回；否则交互询问，通过 python-dotenv 的
    set_key 保存到项目 .env（不存在时创建），并写入 os.environ，
    让当前进程立即能够读取。
    无需密钥的提供商（如 ollama）或未注册的提供商返回 None。"""
    env_var = get_api_key_env(provider)
    if env_var is None:
        return None  # ollama / unknown — no key check possible

    # 密钥可选的提供商（通用 OpenAI 兼容端点或本地服务）
    # 有密钥就读取，但绝不强制弹出交互式密钥输入。
    spec = OPENAI_COMPATIBLE_PROVIDERS.get(provider.lower())
    if spec is not None and spec.key_optional:
        return os.environ.get(env_var)

    existing = os.environ.get(env_var)
    if existing:
        return existing

    if not sys.stdin.isatty():
        console.print(f"[red]{env_var} is not set; there is no terminal to ask for it.[/red]")
        raise typer.Exit(code=1)

    console.print(
        f"\n[yellow]{env_var} is not set in your environment.[/yellow]"
    )
    key = questionary.password(
        f"Paste your {env_var} (will be saved to .env):",
        style=questionary.Style([
            ("text", "fg:cyan"),
            ("highlighted", "noinherit"),
        ]),
    ).ask()
    if not key:
        console.print(
            f"[red]Skipped. API calls will fail until {env_var} is set.[/red]"
        )
        return None

    env_path = find_dotenv(usecwd=True) or str(Path.cwd() / ".env")
    # 该文件保存凭据，写入前必须限制为仅文件所有者可访问。
    # 不存在时以 0600 权限创建，已有文件则收紧权限；set_key 保留文件模式。
    if not os.path.exists(env_path):
        os.close(os.open(env_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
    os.chmod(env_path, 0o600)
    set_key(env_path, env_var, key)
    os.environ[env_var] = key
    console.print(f"[green]Saved {env_var} to {env_path}[/green]")
    return key


def ask_output_language(default=None) -> str:
    """询问报告输出语言。

    仅在 ``default`` 属于菜单内语言时将其预选；
    上次填写的自定义语言是自由文本，菜单无法直接选中。"""
    choices = [
        questionary.Choice("English (default)", "English"),
        questionary.Choice("Chinese (中文)", "Chinese"),
        questionary.Choice("Japanese (日本語)", "Japanese"),
        questionary.Choice("Korean (한국어)", "Korean"),
        questionary.Choice("Hindi (हिन्दी)", "Hindi"),
        questionary.Choice("Spanish (Español)", "Spanish"),
        questionary.Choice("Portuguese (Português)", "Portuguese"),
        questionary.Choice("French (Français)", "French"),
        questionary.Choice("German (Deutsch)", "German"),
        questionary.Choice("Arabic (العربية)", "Arabic"),
        questionary.Choice("Russian (Русский)", "Russian"),
        questionary.Choice("Custom language", "custom"),
    ]
    choice = questionary.select(
        "Select Output Language:",
        choices=choices,
        default=_matching_choice([(c.title, c.value) for c in choices], default),
        style=questionary.Style([
            ("selected", "fg:yellow noinherit"),
            ("highlighted", "fg:yellow noinherit"),
            ("pointer", "fg:yellow noinherit"),
        ]),
    ).ask()

    # 输出语言有合理默认值，因此取消时回退到英语，而不是结束运行；
    # 提供商和模型等必填问题取消时才直接退出。
    if choice is None:
        return "English"
    if choice == "custom":
        return (questionary.text(
            "Enter language name (e.g. Turkish, Vietnamese, Thai, Indonesian):",
            validate=lambda x: len(x.strip()) > 0 or "Please enter a language name.",
        ).ask() or "").strip() or "English"

    return choice
