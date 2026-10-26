"""一次运行的交互式选择：ticker、日期、分析师、研究深度、提供商和模型。"""

import datetime
import os
from pathlib import Path

import typer
from rich.align import Align
from rich.panel import Panel

from cli import updates
from cli.display import (
    console,
)
from cli.prefs import load_last_run, sanitize, save_last_run
from cli.prompts import (
    ask_anthropic_effort,
    ask_gemini_thinking_config,
    ask_glm_region,
    ask_minimax_region,
    ask_openai_reasoning_effort,
    ask_output_language,
    ask_qwen_region,
    confirm_ollama_endpoint,
    detect_asset_type,
    ensure_api_key,
    ensure_signed_in,
    get_ticker,
    parse_analysis_date,
    parse_analysts,
    parse_ticker,
    prompt_openai_compatible_url,
    resolve_backend_url,
    select_analysts,
    select_deep_thinking_agent,
    select_llm_provider,
    select_research_depth,
    select_shallow_thinking_agent,
)
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients.factory import REASONING_EFFORT_PROVIDERS
from tradingagents.llm_clients.openai_client import check_endpoint


def get_user_selections(flags=None):
    """询问本次运行所需配置，并以之前运行的答案作为默认建议。"""
    selections = _prompt_selections(load_last_run(), flags or {})
    save_last_run(selections)
    return selections


def depth_from_env() -> bool:
    """两类讨论轮数都由环境变量指定，因此跳过研究深度提问。"""
    return bool(os.environ.get("TRADINGAGENTS_MAX_DEBATE_ROUNDS")
                and os.environ.get("TRADINGAGENTS_MAX_RISK_ROUNDS"))


def unattended_gaps(flags) -> list[str]:
    """列出无交互终端运行时仍需提供的命令行标志和环境变量。"""
    env = os.environ.get
    gaps = [f"--{name}" for name in ("ticker", "date", "analysts") if flags.get(name) is None]
    gaps += [f"--{name} or --no-{name}" for name in ("save", "show") if flags.get(name) is None]
    if not DEFAULT_CONFIG.get("unified_llm"):
        if not env("TRADINGAGENTS_OUTPUT_LANGUAGE"):
            gaps.append("TRADINGAGENTS_OUTPUT_LANGUAGE")
        if not depth_from_env():
            gaps.append("TRADINGAGENTS_MAX_DEBATE_ROUNDS and TRADINGAGENTS_MAX_RISK_ROUNDS")
        if not env("TRADINGAGENTS_LLM_PROVIDER"):
            gaps.append("TRADINGAGENTS_LLM_PROVIDER")
        if not (env("TRADINGAGENTS_QUICK_THINK_LLM") or env("TRADINGAGENTS_DEEP_THINK_LLM")):
            gaps.append("TRADINGAGENTS_QUICK_THINK_LLM or TRADINGAGENTS_DEEP_THINK_LLM")
    return gaps


def _check_tier_providers(main_provider: str, backend_url: str | None = None) -> None:
    """在执行前检查各模型层级实际使用的提供商（#1440）。
    
    其他提供商所属的模型层级必须从对应变量获取模型，因为交互式模型列表
    只展示主提供商的模型。此处应提前检查每个实际使用的提供商
    是否有可用端点、登录状态和 API Key，并提示补齐缺少的密钥；
    不能等到首次调用时才检查（deep 层级可能很晚才执行，
    届时前面的调用费用已产生）。未被任何层级使用的提供商无需密钥。"""
    # 统一的 LLM_* 配置已经校验端点与密钥；这里无需再询问
    # 旧版提供商专属环境变量或登录信息。
    if DEFAULT_CONFIG.get("unified_llm"):
        return
    used = []
    for tier in ("quick", "deep"):
        provider = (DEFAULT_CONFIG.get(f"{tier}_think_provider") or main_provider).lower()
        on_main = provider == main_provider.lower()
        if not on_main:
            variable = f"TRADINGAGENTS_{tier.upper()}_THINK_LLM"
            if not os.environ.get(variable):
                console.print(f"[red]The {tier} tier runs on {provider}; set {variable} to one of its models.[/red]")
                raise typer.Exit(code=1)
        # 使用 create_tier_client 解析出的模型层级端点。
        url = DEFAULT_CONFIG.get(f"{tier}_think_backend_url") or (backend_url if on_main else None)
        try:
            check_endpoint(provider, url)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from None
        if provider not in used:
            used.append(provider)
    for provider in used:
        ensure_signed_in(provider)
        ensure_api_key(provider)


def _from_flag(parse, value, *args):
    """对命令行标志的值执行与交互输入相同的校验；非法值会终止运行。"""
    try:
        return parse(value, *args)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None


def _prompt_selections(prefs, flags):
    """依次执行配置选择步骤；``prefs`` 用于预填默认值，
    命令行标志和环境变量可直接跳过对应问题。"""
    with open(Path(__file__).parent / "static" / "welcome.txt", encoding="utf-8") as f:
        welcome_ascii = f.read()

    welcome_content = f"{welcome_ascii}\n"
    welcome_content += "[bold green]Trader: Multi-Agent Market Research - CLI[/bold green]\n\n"
    welcome_content += "[bold]Workflow Steps:[/bold]\n"
    welcome_content += "I. Analyst Team → II. Research Team → III. Trader → IV. Risk Management → V. Portfolio Management\n\n"
    welcome_content += (
        "[dim]Trader · derived from [TradingAgents](https://github.com/TauricResearch/TradingAgents)[/dim]"
    )

    welcome_box = Panel(
        welcome_content,
        border_style="green",
        padding=(1, 2),
        title="Welcome to Trader",
        subtitle="Multi-Agents LLM Financial Trading Framework",
    )
    console.print(Align.center(welcome_box))
    console.print()
    console.print()  # Add vertical space before announcements

    # 更新信息来自后台刷新的缓存，且仅面向交互终端用户；
    # 运行流程不会等待缓存刷新，无人值守模式会跳过提示。
    if os.environ.get("TRADER_ENABLE_UPSTREAM_NOTICES") == "1" and updates.enabled():
        updates.show(console)
        updates.start_background_refresh()

    def create_question_box(title, prompt, default=None):
        box_content = f"[bold]{title}[/bold]\n"
        box_content += f"[dim]{prompt}[/dim]"
        if default:
            box_content += f"\n[dim]Default: {default}[/dim]"
        return Panel(box_content, border_style="blue", padding=(1, 2))

    def thinking_value_or_prompt(env_var, config_key, label, box_title, box_body, prompt_fn):
        """返回环境变量配置的推理参数，或通过交互方式询问。
        
        设置 ``env_var`` 后跳过交互式选择，直接使用环境变量
        覆盖到 DEFAULT_CONFIG 的值；这一规则与其他配置步骤的
        环境变量优先级保持一致。"""
        if os.environ.get(env_var):
            value = DEFAULT_CONFIG[config_key]
            console.print(f"[green]✓ {label} from environment:[/green] {value}")
            return value
        console.print(create_question_box(box_title, box_body))
        return prompt_fn()

    # 第 1 步：选择交易标的代码。
    if flags.get("ticker") is not None:
        selected_ticker = _from_flag(parse_ticker, flags["ticker"])
        console.print(f"[green]✓ Ticker from --ticker:[/green] {selected_ticker}")
    else:
        console.print(
            create_question_box(
                "Step 1: Ticker Symbol",
                "Enter the ticker, with exchange suffix when needed (e.g. SPY, 0700.HK, BTC-USD)",
                "SPY",
            )
        )
        selected_ticker = get_ticker()
    asset_type = detect_asset_type(selected_ticker)
    # 仅在不走默认股票路径时显示资产类型，
    # 避免每次运行都重复打印 “stock”。
    if asset_type.value != "stock":
        console.print(
            f"[green]Detected asset type:[/green] {asset_type.value}"
        )

    # 第 2 步：选择分析日期。
    if flags.get("date") is not None:
        analysis_date = _from_flag(parse_analysis_date, flags["date"])
        console.print(f"[green]✓ Analysis date from --date:[/green] {analysis_date}")
    else:
        default_date = datetime.datetime.now().strftime("%Y-%m-%d")
        console.print(
            create_question_box(
                "Step 2: Analysis Date",
                "Enter the analysis date (YYYY-MM-DD)",
                default_date,
            )
        )
        analysis_date = get_analysis_date()

    # 第 3 步：选择输出语言（设置 TRADINGAGENTS_OUTPUT_LANGUAGE 时跳过）。
    if os.environ.get("TRADINGAGENTS_OUTPUT_LANGUAGE") or DEFAULT_CONFIG.get("unified_llm"):
        output_language = DEFAULT_CONFIG["output_language"]
        console.print(
            f"[green]✓ Output language from environment:[/green] {output_language}"
        )
    else:
        console.print(
            create_question_box(
                "Step 3: Output Language",
                "Select the language for analyst reports and final decision"
            )
        )
        output_language = ask_output_language(prefs.get("output_language"))

    # 第 4 步：选择分析师。
    prefs = sanitize(prefs, asset_type.value)
    if flags.get("analysts") is not None:
        selected_analysts = _from_flag(parse_analysts, flags["analysts"], asset_type)
    else:
        console.print(
            create_question_box(
                "Step 4: Analysts Team", "Select your LLM analyst agents for the analysis"
            )
        )
        selected_analysts = select_analysts(asset_type, prefs.get("analysts"))
    console.print(
        f"[green]Selected analysts:[/green] {', '.join(analyst.value for analyst in selected_analysts)}"
    )

    # 第 5 步：选择研究深度（两个讨论轮数均由环境变量设置时跳过）。
    # 研究深度映射到辩论轮数与风险讨论轮数；若通过
    # TRADINGAGENTS_MAX_DEBATE_ROUNDS / _MAX_RISK_ROUNDS
    # 同时指定，则不再交互提问，并保留环境变量的值（#977）。
    if depth_from_env() or DEFAULT_CONFIG.get("unified_llm"):
        selected_research_depth = DEFAULT_CONFIG["max_debate_rounds"]
        console.print(
            f"[green]✓ Research depth from environment:[/green] "
            f"{DEFAULT_CONFIG['max_debate_rounds']} debate / "
            f"{DEFAULT_CONFIG['max_risk_discuss_rounds']} risk rounds"
        )
    else:
        console.print(
            create_question_box(
                "Step 5: Research Depth", "Select your research depth level"
            )
        )
        selected_research_depth = select_research_depth(prefs.get("research_depth"))

    # 第 6 步：选择 LLM 提供商（设置 TRADINGAGENTS_LLM_PROVIDER 时跳过）。
    # 若设置了 TRADINGAGENTS_LLM_BACKEND_URL 则使用该地址，
    # 否则采用提供商默认端点，与菜单选择的默认值保持一致。
    provider_from_env = bool(os.environ.get("TRADINGAGENTS_LLM_PROVIDER")) or bool(DEFAULT_CONFIG.get("unified_llm"))
    if provider_from_env:
        selected_llm_provider = DEFAULT_CONFIG["llm_provider"].lower()
        backend_url = resolve_backend_url(
            selected_llm_provider, env_url=DEFAULT_CONFIG["backend_url"]
        )
        console.print(f"[green]✓ LLM provider from environment:[/green] {selected_llm_provider}")
        console.print(f"[green]✓ Backend URL:[/green] {backend_url}")
    else:
        console.print(
            create_question_box(
                "Step 6: LLM Provider", "Select your LLM provider"
            )
        )
        selected_llm_provider, backend_url = select_llm_provider(prefs.get("llm_provider"))

        # 有地域区分的提供商通过第二个问题选择地区，
        # 以免主下拉菜单过于复杂（中国大陆与国际账户不能共用 API Key）。
        if selected_llm_provider == "qwen":
            selected_llm_provider, backend_url = ask_qwen_region()
        elif selected_llm_provider == "minimax":
            selected_llm_provider, backend_url = ask_minimax_region()
        elif selected_llm_provider == "glm":
            selected_llm_provider, backend_url = ask_glm_region()

        # 即使提供商是交互选择的，也必须尊重显式设置的后端 URL，
        # 防止被菜单默认地址覆盖（#978）。
        backend_url = resolve_backend_url(
            selected_llm_provider, backend_url, env_url=DEFAULT_CONFIG["backend_url"]
        )

        # 通用 OpenAI 兼容端点没有默认地址；
        # 菜单或环境变量都未提供时必须询问用户。
        if selected_llm_provider == "openai_compatible" and not backend_url:
            remembered_url = (prefs.get("backend_url")
                              if prefs.get("llm_provider") == selected_llm_provider else None)
            backend_url = prompt_openai_compatible_url(remembered_url)

        # 选择 Ollama 时，在模型选择之前显示解析后的端点
        # （OLLAMA_BASE_URL 或默认值），让用户知道实际连接位置。
        if selected_llm_provider == "ollama":
            confirm_ollama_endpoint(backend_url)


    _check_tier_providers(selected_llm_provider, backend_url)

    # 第 7 步：选择推理模型（任一模型由环境变量指定时跳过对应选项）。
    if (DEFAULT_CONFIG.get("unified_llm") or os.environ.get("TRADINGAGENTS_QUICK_THINK_LLM")
            or os.environ.get("TRADINGAGENTS_DEEP_THINK_LLM")):
        selected_shallow_thinker = DEFAULT_CONFIG["quick_think_llm"]
        selected_deep_thinker = DEFAULT_CONFIG["deep_think_llm"]
        console.print(
            f"[green]✓ Thinking agents from environment:[/green] "
            f"quick={selected_shallow_thinker}, deep={selected_deep_thinker}"
        )
    else:
        console.print(
            create_question_box(
                "Step 7: Thinking Agents", "Select your thinking agents for analysis"
            )
        )
        remembered = prefs if prefs.get("llm_provider") == selected_llm_provider else {}
        selected_shallow_thinker = select_shallow_thinking_agent(
            selected_llm_provider, remembered.get("quick_think_llm")
        )
        selected_deep_thinker = select_deep_thinking_agent(
            selected_llm_provider, remembered.get("deep_think_llm")
        )

    # 第 8 步：配置提供商专属的推理参数。每个选项都可以通过
    # 对应的 TRADINGAGENTS_* 环境变量设置；变量已设置时
    # （或提供商本身来自环境变量）跳过提问，直接使用配置值。
    # 优先级规则与前面步骤一致；None 表示使用提供商自己的默认值。
    thinking_level = None
    reasoning_effort = None
    anthropic_effort = None

    provider_lower = selected_llm_provider.lower()
    if provider_from_env:
        thinking_level = DEFAULT_CONFIG["google_thinking_level"]
        reasoning_effort = DEFAULT_CONFIG["openai_reasoning_effort"]
        anthropic_effort = DEFAULT_CONFIG["anthropic_effort"]
    elif provider_lower == "google":
        thinking_level = thinking_value_or_prompt(
            "TRADINGAGENTS_GOOGLE_THINKING_LEVEL", "google_thinking_level",
            "Gemini thinking mode", "Step 8: Thinking Mode",
            "Configure Gemini thinking mode", ask_gemini_thinking_config,
        )
    elif provider_lower in REASONING_EFFORT_PROVIDERS:
        reasoning_effort = thinking_value_or_prompt(
            "TRADINGAGENTS_OPENAI_REASONING_EFFORT", "openai_reasoning_effort",
            "Reasoning effort", "Step 8: Reasoning Effort",
            "Configure OpenAI reasoning effort level", ask_openai_reasoning_effort,
        )
    elif provider_lower == "anthropic":
        anthropic_effort = thinking_value_or_prompt(
            "TRADINGAGENTS_ANTHROPIC_EFFORT", "anthropic_effort",
            "Claude effort", "Step 8: Effort Level",
            "Configure Claude effort level", ask_anthropic_effort,
        )

    return {
        "ticker": selected_ticker,
        "asset_type": asset_type.value,
        "analysis_date": analysis_date,
        "analysts": selected_analysts,
        "research_depth": selected_research_depth,
        "llm_provider": selected_llm_provider.lower(),
        "backend_url": backend_url,
        "quick_think_llm": selected_shallow_thinker,
        "deep_think_llm": selected_deep_thinker,
        "google_thinking_level": thinking_level,
        "openai_reasoning_effort": reasoning_effort,
        "anthropic_effort": anthropic_effort,
        "output_language": output_language,
    }


def get_analysis_date():
    """从用户输入中获取分析日期。"""
    while True:
        date_str = typer.prompt(
            "", default=datetime.datetime.now().strftime("%Y-%m-%d")
        )
        try:
            return parse_analysis_date(date_str)
        except ValueError as exc:
            console.print(f"[red]Error: {exc}[/red]")
