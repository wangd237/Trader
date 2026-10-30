# Phase 2 smoke test: real LLM + frozen historical OHLCV (Market Analyst)

This is a **data-isolation test harness**, not an LLM-as-Judge. The real
LangGraph and model calls still execute. No production trading behavior is
changed unless the explicit benchmark configuration is enabled.

## Prepare the local CSV

Obtain genuine daily historical NVDA OHLCV from a source you can audit
(e.g. an authorized data export). **No market prices are invented or
bundled by this repository.** Required columns are:

    Date,Open,High,Low,Close,Volume
    (one daily observation per row, ISO YYYY-MM-DD dates)

For the default experiment date 2026-09-30, use **at least 200 daily
trading rows up to that date**, with the latest row within seven calendar
days of the analysis date. Prefer substantially more than 200 rows so
the 200-day SMA can be computed from sufficient history. Values must be
real, nonnegative volume, and OHLC price bounds must be consistent.
Preserve the source and adjustment policy (adjusted/unadjusted prices)
with the CSV. Do not combine bars from different adjustment conventions.

Do not commit API keys or unlicensed market datasets. Test-generated CSV
fixtures in `tests/test_frozen_market_baseline.py` are **synthetic and not
valid historical prices**.

## Run

Pull main after this PR merges and install the local package:

    python -m pip install -e ".[dev]"
    python -m pytest -q tests/test_frozen_market_baseline.py

Put LLM_API_KEY, LLM_MODEL and LLM_BASE_URL in your root .env file
as documented in README. On Windows PowerShell:

    python scripts/run_real_baseline.py --csv "E:\\market_data\\NVDA.csv" --source "provider and export date"

Optional: --trade-date 2026-09-30 --attempt 02 --max-tool-rounds 3

It saves a manifest containing SHA-256, coverage, ticker, model identifiers,
source description, and experiment settings, alongside the existing
`trace.jsonl` and `evaluation.json` files:

    results/trajectory_baseline/NVDA_2026-09-30_market/01/

The script validates data and pins its SHA-256 **before making a paid
LLM call**. The same digest is recorded in trace metadata for A/B runs.

## Isolation and safety contract

- `get_stock_data`: slices the locally pinned CSV for the requested date
  range, rejects unsupported date windows instead of returning unrelated data.
- `get_indicators`: uses the existing stockstats implementation against
  the same frozen CSV. No Yahoo download.
- `get_verified_market_snapshot`: reuses existing deterministic snapshot
  logic and the same frozen CSV.
- Benchmark ticker identity is pinned and does not query live Yahoo.
- Unsupported data tools are blocked in this Market-only experiment.
- No falling back to Yahoo/Alpha Vantage under a missing, modified or stale
  fixture. The SHA check is per access (slow but unambiguous for this smoke).
- The Memory Log location is isolated per attempt. Use fresh attempt IDs.
- No full raw tool results or full prompt history are stored; this **cannot**
  yet score Handoff or Tool Necessity semantically.

Do not treat a structurally passing trajectory as proof of financial validity.
LLM-as-Judge and broader multi-agent data freezing are separate work.
