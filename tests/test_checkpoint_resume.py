"""检查点恢复测试：分析中途崩溃后，再次运行应从上次成功节点继续。"""

import tempfile
import unittest
from typing import TypedDict

from langgraph.graph import END, StateGraph

from tradingagents.graph.checkpointer import (
    checkpoint_step,
    clear_checkpoint,
    get_checkpointer,
    thread_id,
)

# 可变标志：在首次运行时模拟崩溃。
_should_crash = False


class _SimpleState(TypedDict):
    count: int


def _node_a(state: _SimpleState) -> dict:
    return {"count": state["count"] + 1}


def _node_b(state: _SimpleState) -> dict:
    if _should_crash:
        raise RuntimeError("simulated mid-analysis crash")
    return {"count": state["count"] + 10}


def _build_graph() -> StateGraph:
    builder = StateGraph(_SimpleState)
    builder.add_node("analyst", _node_a)
    builder.add_node("trader", _node_b)
    builder.set_entry_point("analyst")
    builder.add_edge("analyst", "trader")
    builder.add_edge("trader", END)
    return builder


class TestCheckpointResume(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ticker = "TEST"
        self.date = "2026-04-20"

    def test_crash_and_resume(self):
        """在 trader 节点模拟崩溃，随后从检查点恢复。"""
        global _should_crash
        builder = _build_graph()
        tid = thread_id(self.ticker, self.date)
        cfg = {"configurable": {"thread_id": tid}}

        # 首次运行：在 Trader 节点崩溃。
        _should_crash = True
        with get_checkpointer(self.tmpdir, self.ticker) as saver:
            graph = builder.compile(checkpointer=saver)
            with self.assertRaises(RuntimeError):
                graph.invoke({"count": 0}, config=cfg)

        # 分析师完成后应在步骤 1 留下检查点。
        self.assertIsNotNone(checkpoint_step(self.tmpdir, self.ticker, self.date))
        step = checkpoint_step(self.tmpdir, self.ticker, self.date)
        self.assertEqual(step, 1)

        # 第二次运行：从检查点恢复，Trader 此次成功。
        _should_crash = False
        with get_checkpointer(self.tmpdir, self.ticker) as saver:
            graph = builder.compile(checkpointer=saver)
            result = graph.invoke(None, config=cfg)

        # 分析师增加 1，Trader 增加 10，合计 11。
        self.assertEqual(result["count"], 11)

    def test_clear_checkpoint_allows_fresh_start(self):
        """清除检查点之后，Graph 应从头开始运行。"""
        global _should_crash
        builder = _build_graph()
        tid = thread_id(self.ticker, self.date)
        cfg = {"configurable": {"thread_id": tid}}

        # 通过模拟崩溃生成检查点。
        _should_crash = True
        with get_checkpointer(self.tmpdir, self.ticker) as saver:
            graph = builder.compile(checkpointer=saver)
            with self.assertRaises(RuntimeError):
                graph.invoke({"count": 0}, config=cfg)

        self.assertIsNotNone(checkpoint_step(self.tmpdir, self.ticker, self.date))

        # 清理检查点。
        clear_checkpoint(self.tmpdir, self.ticker, self.date)
        self.assertIsNone(checkpoint_step(self.tmpdir, self.ticker, self.date))

        # 新运行应从头开始并成功。
        _should_crash = False
        with get_checkpointer(self.tmpdir, self.ticker) as saver:
            graph = builder.compile(checkpointer=saver)
            result = graph.invoke({"count": 0}, config=cfg)

        self.assertEqual(result["count"], 11)

    def test_different_date_starts_fresh(self):
        """不同分析日期绝不能复用已有检查点。"""
        global _should_crash
        builder = _build_graph()
        date2 = "2026-04-21"

        # 使用 date1 运行并崩溃，留下检查点。
        _should_crash = True
        tid1 = thread_id(self.ticker, self.date)
        with get_checkpointer(self.tmpdir, self.ticker) as saver:
            graph = builder.compile(checkpointer=saver)
            with self.assertRaises(RuntimeError):
                graph.invoke({"count": 0}, config={"configurable": {"thread_id": tid1}})

        self.assertIsNotNone(checkpoint_step(self.tmpdir, self.ticker, self.date))

        # date2 不应存在相应检查点。
        self.assertIsNone(checkpoint_step(self.tmpdir, self.ticker, date2))

        # 使用 date2 运行，应全新启动并成功。
        _should_crash = False
        tid2 = thread_id(self.ticker, date2)
        self.assertNotEqual(tid1, tid2)

        with get_checkpointer(self.tmpdir, self.ticker) as saver:
            graph = builder.compile(checkpointer=saver)
            result = graph.invoke({"count": 0}, config={"configurable": {"thread_id": tid2}})

        # 新运行：分析师 +1，Trader +10，最终为 11。
        self.assertEqual(result["count"], 11)

        # 原日期的检查点必须保持原样。
        self.assertIsNotNone(checkpoint_step(self.tmpdir, self.ticker, self.date))


class TestCheckpointSignature(unittest.TestCase):
    """分析师选择、研究深度或资产模式改变后，图结构不同，
    不能恢复旧运行的检查点（#1089）。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ticker = "TEST"
        self.date = "2026-04-20"

    def test_empty_signature_is_legacy_id(self):
        self.assertEqual(
            thread_id(self.ticker, self.date),
            thread_id(self.ticker, self.date, ""),
        )

    def test_signature_changes_thread_id(self):
        legacy = thread_id(self.ticker, self.date)
        sig_a = thread_id(self.ticker, self.date, "analysts=market,news|asset=stock")
        sig_b = thread_id(self.ticker, self.date, "analysts=market|asset=stock")
        self.assertNotEqual(sig_a, sig_b)          # different graph shapes differ
        self.assertNotEqual(legacy, sig_a)         # signature-keyed differs from legacy
        self.assertEqual(                          # same inputs are stable
            sig_a, thread_id(self.ticker, self.date, "analysts=market,news|asset=stock")
        )

    def test_different_signature_starts_fresh(self):
        global _should_crash
        builder = _build_graph()
        sig1 = "analysts=market,news,fundamentals|asset=stock"
        sig2 = "analysts=market|asset=stock"       # dropped analysts -> different graph

        _should_crash = True
        tid1 = thread_id(self.ticker, self.date, sig1)
        with get_checkpointer(self.tmpdir, self.ticker) as saver:
            graph = builder.compile(checkpointer=saver)
            with self.assertRaises(RuntimeError):
                graph.invoke({"count": 0}, config={"configurable": {"thread_id": tid1}})

        self.assertIsNotNone(checkpoint_step(self.tmpdir, self.ticker, self.date, sig1))
        # 不同图结构不应存在可供恢复的对应检查点。
        self.assertIsNone(checkpoint_step(self.tmpdir, self.ticker, self.date, sig2))

        _should_crash = False
        tid2 = thread_id(self.ticker, self.date, sig2)
        self.assertNotEqual(tid1, tid2)
        with get_checkpointer(self.tmpdir, self.ticker) as saver:
            graph = builder.compile(checkpointer=saver)
            result = graph.invoke({"count": 0}, config={"configurable": {"thread_id": tid2}})
        self.assertEqual(result["count"], 11)
        # sig1 对应的检查点应保持不变。
        self.assertIsNotNone(checkpoint_step(self.tmpdir, self.ticker, self.date, sig1))

    def test_a_resume_under_other_settings_starts_fresh(self):
        """报告绑定了运行使用的提供商、模型组合、语言及数据源链，
        恢复时不能复用其他配置生成的报告。"""
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        g = object.__new__(TradingAgentsGraph)
        g.selected_analysts = ("market", "news")
        base_config = {"max_debate_rounds": 1, "max_risk_discuss_rounds": 1, "llm_provider": "openai",
                       "quick_think_llm": "q", "deep_think_llm": "d", "output_language": "English",
                       "data_vendors": {"core_stock_apis": "yfinance"}, "tool_vendors": {}}
        g.config = dict(base_config)
        base = g._run_signature("stock")
        for key, value in (("llm_provider", "google"), ("quick_think_llm", "q2"), ("deep_think_llm", "d2"),
                           ("output_language", "Deutsch"),
                           ("data_vendors", {"core_stock_apis": "alpha_vantage"}),
                           ("tool_vendors", {"get_news": "alpha_vantage"}),
                           ("backend_url", "http://other-endpoint/v1"), ("max_tool_rounds", 30),
                           ("temperature", 0.2), ("openai_reasoning_effort", "high")):
            g.config = {**base_config, key: value}
            self.assertNotEqual(base, g._run_signature("stock"), key)
        # 文件保存位置和重试方式不影响运行所生成的内容。
        for key, value in (("results_dir", "/elsewhere"), ("data_cache_dir", "/cache"),
                           ("memory_log_path", "/log.md"), ("checkpoint_enabled", True), ("llm_max_retries", 9)):
            g.config = {**base_config, key: value}
            self.assertEqual(base, g._run_signature("stock"), key)
        g.config = dict(base_config)
        self.assertEqual(base, g._run_signature("stock"))

    def test_run_signature_captures_graph_shape(self):
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        # 构造轻量实例，直接测试纯辅助方法，跳过复杂的 __init__。
        g = object.__new__(TradingAgentsGraph)
        g.selected_analysts = ("market", "news")
        g.config = {"max_debate_rounds": 1, "max_risk_discuss_rounds": 1}
        base = g._run_signature("stock")

        self.assertNotEqual(base, g._run_signature("crypto"))     # asset mode
        g.selected_analysts = ("market",)
        self.assertNotEqual(base, g._run_signature("stock"))      # analyst selection
        g.selected_analysts = ("market", "news")
        g.config = {"max_debate_rounds": 3, "max_risk_discuss_rounds": 1}
        self.assertNotEqual(base, g._run_signature("stock"))      # debate depth
        g.config = {"max_debate_rounds": 1, "max_risk_discuss_rounds": 5}
        self.assertNotEqual(base, g._run_signature("stock"))      # risk depth
        # 相同输入应产生稳定一致的结果。
        g.config = {"max_debate_rounds": 1, "max_risk_discuss_rounds": 1}
        self.assertEqual(base, g._run_signature("stock"))
        # 串行拓扑中生成的检查点不能被并行拓扑恢复：
        # 旧检查点可能留下新图中不存在的待执行节点，导致汇合节点永不触发。
        self.assertIn("analysts=parallel", base)


if __name__ == "__main__":
    unittest.main()
