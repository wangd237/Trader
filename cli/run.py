"""从 CLI 运行一次分析：构建图、将执行过程流式展示在实时视图中，并保存报告。"""

import os
import sys
import time
import webbrowser
from functools import wraps
from pathlib import Path

import typer
from rich.live import Live

from cli.display import (
    ANALYST_ORDER,
    AnalystWallTimeTracker,
    classify_message_type,
    console,
    create_layout,
    display_complete_report,
    message_buffer,
    update_analyst_statuses,
    update_display,
    update_research_team_status,
)
from cli.selections import depth_from_env, get_user_selections, unattended_gaps
from cli.stats_handler import StatsCallbackHandler
from tradingagents.agents.rating import is_review, run_rating
from tradingagents.dataflows.symbols import safe_ticker_component
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.analyst_execution import (
    build_analyst_execution_plan,
)
from tradingagents.graph.trading_graph import TradingAgentsGraph


def _run_directory(config: dict, ticker: str, trade_date: str) -> Path:
    """确定本次运行的输出目录，并校验 ticker 是否可安全用作路径组件。
    
    其他拼接 ticker 的路径也会先做校验；例如 ``..`` 可能使输出落到结果目录之外。"""
    return Path(config["results_dir"]) / safe_ticker_component(ticker) / trade_date


def _announce_checkpoint_state(graph, ticker: str, trade_date: str) -> None:
    """在用户可见的位置说明本次运行是从已保存的检查点恢复，还是全新启动。
    
    Graph 虽然会记录这一信息，但 CLI 没有配置日志输出，且实时视图占用了终端屏幕，因此之前用户无法看到恢复提示。"""
    if getattr(graph, "_resuming", False):
        message_buffer.add_message(
            "System", f"Resuming the saved run for {ticker} on {trade_date}"
        )
    else:
        message_buffer.add_message("System", f"Starting fresh for {ticker} on {trade_date}")


def _build_run_config(selections: dict, checkpoint: bool | None) -> dict:
    """根据交互式选择组装运行配置，同时遵守环境变量的优先级。
    
    轮数和检查点选项遵循“显式环境变量或命令行标志优先”：除非用户通过 CLI 覆盖，否则保留 DEFAULT_CONFIG 中已由环境变量设置的值。"""
    config = DEFAULT_CONFIG.copy()
    # 研究深度决定两类讨论的轮数，但显式设置的环境变量
    # (TRADINGAGENTS_MAX_DEBATE_ROUNDS / _MAX_RISK_ROUNDS)
    # 优先于交互式选择，因此保留环境变量设置的值（#977）。
    rounds = (("TRADINGAGENTS_MAX_DEBATE_ROUNDS", "max_debate_rounds"),
              ("TRADINGAGENTS_MAX_RISK_ROUNDS", "max_risk_discuss_rounds"))
    depth_was_asked = not depth_from_env()
    for env_var, key in rounds:
        if not os.environ.get(env_var):
            config[key] = selections["research_depth"]
        elif depth_was_asked:
            # 只有两个轮数都由环境变量设置时才跳过深度提问；
            # 既然显示了该问题，就应提示用户哪一个轮数被环境变量覆盖。
            console.print(
                f"[green]✓ {key} from environment:[/green] {config[key]} "
                f"(set by {env_var}, so the research depth you chose does not apply to it)"
            )
    config["quick_think_llm"] = selections["quick_think_llm"]
    config["deep_think_llm"] = selections["deep_think_llm"]
    config["backend_url"] = selections["backend_url"]
    config["llm_provider"] = selections["llm_provider"].lower()
    # 仅询问主 LLM 提供商的推理配置；其他提供商所属层级
    # 仍保留各自已配置的值（#1440）。
    for key in ("google_thinking_level", "openai_reasoning_effort", "anthropic_effort"):
        if selections.get(key) is not None:
            config[key] = selections[key]
    config["output_language"] = selections.get("output_language", "English")
    # 只有显式传入 --checkpoint/--no-checkpoint 才覆盖设置；
    # 省略标志时保留 TRADINGAGENTS_CHECKPOINT_ENABLED 或默认值（#976）。
    if checkpoint is not None:
        config["checkpoint_enabled"] = checkpoint
    return config


def run_analysis(checkpoint: bool | None = None, portfolio=None, flags=None):
    flags = flags or {}
    # 没有交互终端就无法回答提问；应在调用模型前列出所有
    # 尚未提供的必填选择，而不是遇到第一个缺失项就停止。
    if not (sys.stdin and sys.stdin.isatty()):
        gaps = unattended_gaps(flags)
        if gaps:
            console.print("[red]No terminal to answer the setup questions. Set:[/red]")
            for gap in gaps:
                console.print(f"  {gap}")
            raise typer.Exit(code=1)

    selections = get_user_selections(flags)

    config = _build_run_config(selections, checkpoint)

    stats_handler = StatsCallbackHandler()

    # 将分析师选择规范化为预定义顺序（选择本身视为集合，但执行顺序固定）。
    selected_set = {analyst.value for analyst in selections["analysts"]}
    selected_analyst_keys = [a for a in ANALYST_ORDER if a in selected_set]
    analyst_execution_plan = build_analyst_execution_plan(selected_analyst_keys)
    analyst_wall_time_tracker = AnalystWallTimeTracker(analyst_execution_plan)

    graph = TradingAgentsGraph(
        selected_analyst_keys,
        config=config,
        debug=True,
        callbacks=[stats_handler],
    )

    message_buffer.init_for_analysis(selected_analyst_keys)

    # 记录开始时间，用于显示已耗时。
    start_time = time.time()

    results_dir = _run_directory(config, selections["ticker"], selections["analysis_date"])
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir = results_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / "message_tool.log"
    log_file.touch(exist_ok=True)

    def save_message_decorator(obj, func_name):
        func = getattr(obj, func_name)

        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, message_type, content = obj.messages[-1]
            content = content.replace("\n", " ")  # Replace newlines with spaces
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} [{message_type}] {content}\n")
        return wrapper

    def save_tool_call_decorator(obj, func_name):
        func = getattr(obj, func_name)

        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, tool_name, args = obj.tool_calls[-1]
            args_str = ", ".join(f"{k}={v}" for k, v in args.items())
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} [Tool Call] {tool_name}({args_str})\n")
        return wrapper

    def save_report_section_decorator(obj, func_name):
        func = getattr(obj, func_name)

        @wraps(func)
        def wrapper(section_name, content):
            func(section_name, content)
            if section_name in obj.report_sections and obj.report_sections[section_name] is not None:
                content = obj.report_sections[section_name]
                if content:
                    file_name = f"{section_name}.md"
                    text = "\n".join(str(item) for item in content) if isinstance(content, list) else content
                    with open(report_dir / file_name, "w", encoding="utf-8") as f:
                        f.write(text)
        return wrapper

    message_buffer.add_message = save_message_decorator(message_buffer, "add_message")
    message_buffer.add_tool_call = save_tool_call_decorator(message_buffer, "add_tool_call")
    message_buffer.update_report_section = save_report_section_decorator(message_buffer, "update_report_section")

    layout = create_layout()

    # 使用备用屏幕，避免布局超过窗口高度时因滚动重复重绘；
    # 退出该区域后，在正常屏幕上打印最终报告。
    with Live(layout, refresh_per_second=4, screen=True):
        # 初始化显示。
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        message_buffer.add_message("System", f"Selected ticker: {selections['ticker']}")
        if selections["asset_type"] != "stock":
            message_buffer.add_message("System", f"Detected asset type: {selections['asset_type']}")
        message_buffer.add_message(
            "System", f"Analysis date: {selections['analysis_date']}"
        )
        message_buffer.add_message(
            "System",
            f"Selected analysts: {', '.join(analyst.value for analyst in selections['analysts'])}",
        )
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # 各分析师同时开始执行。
        for spec in analyst_execution_plan.specs:
            message_buffer.update_agent_status(spec.agent_node, "in_progress")
            analyst_wall_time_tracker.mark_started(spec.key)
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        spinner_text = (
            f"Analyzing {selections['ticker']} on {selections['analysis_date']}..."
        )
        update_display(layout, spinner_text, stats_handler=stats_handler, start_time=start_time)

        # 使用与 propagate() 相同的初始状态，并包含解析后的标的身份；
        # Graph 的 Memory Log 步骤与分析师并行结算历史决策，
        # 同时加载历史经验。
        init_agent_state = graph.create_run_state(
            selections["ticker"], selections["analysis_date"], selections["asset_type"], portfolio
        )
        # 向 Graph 配置传入回调，以跟踪工具执行；
        # LLM 调用跟踪则在 LLM 构造函数中单独处理。
        args = graph.propagator.get_graph_args(callbacks=[stats_handler])

        # 重新编译并接入 checkpointer，同时注入 thread_id，
        # 让 CLI 路径上的 --checkpoint 真正支持保存和恢复（#1249）；
        # 禁用检查点时不会产生影响，并在后面的 finally 中清理。
        checkpoint_tid = graph.begin_checkpoint(
            selections["ticker"], selections["analysis_date"], selections["asset_type"], portfolio
        )
        if checkpoint_tid is not None:
            args.setdefault("config", {}).setdefault("configurable", {})["thread_id"] = checkpoint_tid
            _announce_checkpoint_state(graph, selections["ticker"], selections["analysis_date"])

        # 流式执行分析；恢复时传入 None，使 LangGraph 从中断处继续，
        # 而不是再次追加初始状态（#1249）。即使流执行报错，
        # try/finally 也会清理 checkpointer。
        trace = []
        try:
            for messages, chunk in graph.stream_run(graph.checkpoint_input(init_agent_state), **args):
                for message in messages:
                    msg_id = getattr(message, "id", None)
                    if msg_id is not None:
                        if msg_id in message_buffer._processed_message_ids:
                            continue
                        message_buffer._processed_message_ids.add(msg_id)

                    msg_type, content = classify_message_type(message)
                    if content and content.strip():
                        message_buffer.add_message(msg_type, content)

                    if hasattr(message, "tool_calls") and message.tool_calls:
                        for tool_call in message.tool_calls:
                            if isinstance(tool_call, dict):
                                message_buffer.add_tool_call(tool_call["name"], tool_call["args"])
                            else:
                                message_buffer.add_tool_call(tool_call.name, tool_call.args)

                if chunk is None:   # a step inside an analyst's graph: messages only
                    update_display(layout, stats_handler=stats_handler, start_time=start_time)
                    continue

                update_analyst_statuses(
                    message_buffer,
                    chunk,
                    wall_time_tracker=analyst_wall_time_tracker,
                )

                # 研究团队：处理投资辩论状态。
                if chunk.get("investment_debate_state"):
                    debate_state = chunk["investment_debate_state"]
                    bull_hist = debate_state.get("bull_history", "").strip()
                    bear_hist = debate_state.get("bear_history", "").strip()
                    judge = (chunk.get("investment_plan") or "").strip()

                    # 仅在确实有内容时更新状态。
                    if bull_hist or bear_hist:
                        update_research_team_status("in_progress")
                    if bull_hist:
                        message_buffer.update_report_section(
                            "investment_plan", f"### Bull Researcher Analysis\n{bull_hist}"
                        )
                    if bear_hist:
                        message_buffer.update_report_section(
                            "investment_plan", f"### Bear Researcher Analysis\n{bear_hist}"
                        )
                    if judge:
                        message_buffer.update_report_section(
                            "investment_plan", f"### Research Manager Decision\n{judge}"
                        )
                        update_research_team_status("completed")
                        message_buffer.update_agent_status("Trader", "in_progress")

                # 交易团队。
                if chunk.get("trader_investment_plan"):
                    message_buffer.update_report_section(
                        "trader_investment_plan", chunk["trader_investment_plan"]
                    )
                    if message_buffer.agent_status.get("Trader") != "completed":
                        message_buffer.update_agent_status("Trader", "completed")
                        message_buffer.update_agent_status("Aggressive Analyst", "in_progress")

                # 风控团队：处理风险辩论状态。
                if chunk.get("risk_debate_state"):
                    risk_state = chunk["risk_debate_state"]
                    agg_hist = risk_state.get("aggressive_history", "").strip()
                    con_hist = risk_state.get("conservative_history", "").strip()
                    neu_hist = risk_state.get("neutral_history", "").strip()
                    judge = (chunk.get("final_trade_decision") or "").strip()

                    if agg_hist:
                        if message_buffer.agent_status.get("Aggressive Analyst") != "completed":
                            message_buffer.update_agent_status("Aggressive Analyst", "in_progress")
                        message_buffer.update_report_section(
                            "final_trade_decision", f"### Aggressive Analyst Analysis\n{agg_hist}"
                        )
                    if con_hist:
                        if message_buffer.agent_status.get("Conservative Analyst") != "completed":
                            message_buffer.update_agent_status("Conservative Analyst", "in_progress")
                        message_buffer.update_report_section(
                            "final_trade_decision", f"### Conservative Analyst Analysis\n{con_hist}"
                        )
                    if neu_hist:
                        if message_buffer.agent_status.get("Neutral Analyst") != "completed":
                            message_buffer.update_agent_status("Neutral Analyst", "in_progress")
                        message_buffer.update_report_section(
                            "final_trade_decision", f"### Neutral Analyst Analysis\n{neu_hist}"
                        )
                    if judge and message_buffer.agent_status.get("Portfolio Manager") != "completed":
                        message_buffer.update_agent_status("Portfolio Manager", "in_progress")
                        message_buffer.update_report_section(
                            "final_trade_decision", f"### Portfolio Manager Decision\n{judge}"
                        )
                        message_buffer.update_agent_status("Aggressive Analyst", "completed")
                        message_buffer.update_agent_status("Conservative Analyst", "completed")
                        message_buffer.update_agent_status("Neutral Analyst", "completed")
                        message_buffer.update_agent_status("Portfolio Manager", "completed")

                update_display(layout, stats_handler=stats_handler, start_time=start_time)

                trace.append(chunk)

            # 流式返回的 chunk 是各节点的状态增量，而非完整状态；
            # 需要合并增量，确保报告包含本次运行生成的全部字段。
            final_state = {}
            for chunk in trace:
                final_state.update(chunk)

            # 运行正常结束时先记录决策，再删除本次检查点，
            # 保证下次从头运行；若中途失败则跳过这两步，
            # 保留检查点以便恢复。
            graph.record_decision(selections["ticker"], selections["analysis_date"], final_state)
            graph.clear_checkpoint_on_success(
                selections["ticker"], selections["analysis_date"], selections["asset_type"], portfolio
            )
        finally:
            # 无论执行成功还是失败，都恢复未启用检查点的普通 Graph。
            graph.end_checkpoint()

        for agent in message_buffer.agent_status:
            message_buffer.update_agent_status(agent, "completed")

        message_buffer.add_message(
            "System", f"Completed analysis for {selections['analysis_date']}"
        )
        message_buffer.add_message("System", analyst_wall_time_tracker.format_summary())

        for section in message_buffer.report_sections:
            if section in final_state:
                message_buffer.update_report_section(section, final_state[section])

        update_display(layout, stats_handler=stats_handler, start_time=start_time)

    # 分析后的交互提问放在 Live 上下文之外，避免界面冲突。
    console.print("\n[bold cyan]Analysis Complete![/bold cyan]\n")

    # 如果决策不可读取，就不能将其视作有效持仓建议；
    # 应明确告知用户，而不是让运行结果看起来一切正常。
    if is_review(run_rating(final_state)):
        console.print(
            "[yellow]No rating could be read from the final decision, so this run "
            "is recorded for review rather than as a position. Re-run, or read the "
            "decision text below and judge it yourself.[/yellow]\n"
        )
    console.print(f"[dim]{analyst_wall_time_tracker.format_summary()}[/dim]")

    _offer_reports(final_state, graph, selections["ticker"],
                   save=flags.get("save"), show=flags.get("show"), html=flags.get("html"))


def _yes(question: str) -> bool:
    return typer.prompt(question, default="Y").strip().upper() in ("Y", "YES", "")


def _graphical_browser():
    """返回能够在本机独立窗口中打开网页的浏览器；找不到时返回 None。
    
    通过 SSH 访问时，页面位于远程机器；lynx、w3m、elinks 等终端浏览器还会占用当前终端，因此不提供这些选项。"""
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
        return None
    try:
        browser = webbrowser.get()
    except webbrowser.Error:
        return None
    if type(browser) is webbrowser.GenericBrowser or isinstance(browser, webbrowser.Elinks):
        return None
    return browser


def _open_page(page: Path) -> None:
    """询问是否在浏览器窗口中打开已保存的页面；若打开失败，则告知页面保存位置。"""
    browser = _graphical_browser()
    if browser is None or not _yes("Open it in your browser?"):
        return
    try:
        opened = browser.open(page.as_uri())
    except (webbrowser.Error, OSError):
        opened = False
    if not opened:
        console.print(f"  [dim]Could not open a browser; the page is at:[/dim] {page}")


def _offer_reports(final_state, graph, ticker, save=None, show=None, html=None):
    """保存并展示报告目录；传入 ``save``、``show``、``html`` 时使用对应参数回答交互问题。
    
    除非 ``html`` 为 False，否则保存的报告包含 HTML 页面。若用户在交互提示中选择保存，还会询问网页相关选项并提供打开入口；若命令行标志已决定是否保存，则不再询问。"""
    asked = save is None
    if asked:
        save = typer.prompt("Save report?", default="Y").strip().upper() in ("Y", "YES", "")
    if save:
        # 报告应保存在 results_dir 而不是工作目录：Docker 的工作目录
        # 位于容器内部，保存到那里会让报告留在容器中；results_dir
        # 则是本次运行其他输出已经使用的挂载目录。
        save_path = graph.default_report_path(ticker)
        if asked:   # someone at the prompt may pick another folder
            save_path = Path(typer.prompt(
                "Save path (press Enter for default)", default=str(save_path)
            ).strip())
        if html is None:
            html = _yes("Also save it as an HTML page?") if asked else True
        saved = False
        try:
            report_file = graph.save_reports(final_state, ticker, save_path, html=html)
            saved = True
            console.print(f"\n[green]✓ Report saved to:[/green] {save_path.resolve()}")
            console.print(f"  [dim]Complete report:[/dim] {report_file.name}")
        except Exception as e:
            console.print(f"[red]Error saving report: {e}[/red]")
        if saved and html:
            page = (save_path / "complete_report.html").resolve()
            console.print(f"  [dim]HTML report:[/dim] {page.name}")
            if asked:
                _open_page(page)

    if show is None:
        show = typer.prompt("\nDisplay full report on screen?", default="Y").strip().upper() in ("Y", "YES", "")
    if show:
        display_complete_report(final_state)
