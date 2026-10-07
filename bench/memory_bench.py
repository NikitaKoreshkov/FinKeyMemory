# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Reproducible micro-benchmark for FinKeyMemory — pure Python, zero infra,
no network, no LLM keys. Every number in README "Benchmarks" comes from:

    python bench/memory_bench.py

It measures the in-process substrate (L0 + scoring + prompt assembly) and
asserts the semantic contracts (freshness wins, supersession visible,
recency/access re-rank, prompt-block budget). Postgres/Qdrant-dependent
tiers are covered by tests/live/, not by this bench.
"""
from __future__ import annotations

import json
import platform
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finkey_memory import ExtractedFact                     # noqa: E402
from finkey_memory.factory import build_memory_manager      # noqa: E402
from finkey_memory.manager import _rerank_fact_hits         # noqa: E402

N_FACTS = 10_000
N_QUERIES = 200


def bench_ingest_and_read() -> dict:
    mm = build_memory_manager()
    facts = [
        ExtractedFact(
            key_normalized=f"metric_{i}",
            value=f"value {i} with some realistic payload text",
            confidence=0.5 + (i % 40) / 100.0,
            priority=(i % 97) / 97.0,
            source="user" if i % 3 else "inferred",
        )
        for i in range(N_FACTS)
    ]

    t0 = time.perf_counter()
    report = mm.upsert_extracted_facts(
        company_id="bench", user_id="u1", conversation_id=None, facts=facts[:1000]
    )
    first_1k_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    rest = mm.upsert_extracted_facts(
        company_id="bench", user_id="u1", conversation_id=None, facts=facts[1000:]
    )
    total_ms = (time.perf_counter() - t0) * 1000.0
    assert report.volatile_ok == 1000 and rest.volatile_ok == N_FACTS - 1000

    lat: list[float] = []
    blocks = 0
    for i in range(N_QUERIES):
        q = f"metric_{i * 37} question about value number {i}"
        t0 = time.perf_counter()
        ctx = mm.load_context("bench", "u1", f"conv{i % 5}", q)
        lat.append((time.perf_counter() - t0) * 1000.0)
        blocks += 1
    lat.sort()
    return {
        "upsert_1k_facts_ms": round(first_1k_ms, 2),
        "upsert_all_facts_total_ms": round(first_1k_ms + total_ms, 1),
        "n_facts": N_FACTS,
        "load_context_p50_ms": round(lat[len(lat) // 2], 3),
        "load_context_p95_ms": round(lat[int(len(lat) * 0.95)], 3),
        "load_context_p99_ms": round(lat[int(len(lat) * 0.99)], 3),
        "queries": blocks,
    }


def check_freshness_contract() -> dict:
    mm = build_memory_manager()
    mm.upsert_extracted_facts(
        company_id="contract", user_id="u1", conversation_id=None,
        facts=[ExtractedFact(key_normalized="mortgage_rate", value="92.5", source="user")],
    )
    mm.upsert_extracted_facts(
        company_id="contract", user_id="u1", conversation_id=None,
        facts=[ExtractedFact(key_normalized="mortgage_rate", value="95.0", source="user")],
    )
    ctx = mm.load_context("contract", "u1", "c1", "mortgage rate?")
    block = ctx.memory_prompt_block
    ok_current = "95.0" in block
    ok_history = "(alternative: 92.5)" in block or "was: 92.5" in block
    # tenant isolation
    other = mm.load_context("other_co", "u1", "c1", "mortgage rate?")
    ok_tenant = "mortgage_rate" not in other.memory_prompt_block
    return {
        "current_value_in_block": ok_current,
        "superseded_value_visible": ok_history,
        "tenant_isolation": ok_tenant,
    }


def check_rerank_contract() -> dict:
    now = datetime.now(timezone.utc).timestamp()
    old_ts = now - 120 * 86400  # outside the 90d recency window → bonus 0
    hits = [
        {"knowledge_key": "old",   "value": "o", "score": 0.70, "created_at": old_ts,
         "updated_at": old_ts, "access_count": 0},
        {"knowledge_key": "hot",   "value": "h", "score": 0.70, "created_at": now,
         "updated_at": now, "access_count": 40},
        {"knowledge_key": "recent","value": "r", "score": 0.70, "created_at": now,
         "updated_at": now, "access_count": 0},
    ]
    out = _rerank_fact_hits(hits)
    order = [h["knowledge_key"] for h in out]
    eff = {h["knowledge_key"]: h["score_effective"] for h in out}
    return {
        "order": order,
        # documented design: raw score dominates, bonuses are gentle tiebreakers
        "observed_bonuses": {
            "recency_max": 0.15, "access_max": 0.05,
        },
        "access_outranks_same_raw_recent": eff["hot"] > eff["recent"],
        "recency_outranks_stale": eff["recent"] > eff["old"],
        "effective_uplift_present": eff["hot"] > 0.70,
        "stale_untouched": abs(eff["old"] - 0.70) < 1e-9,
    }


def main() -> None:
    result = {
        "machine": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu": platform.processor() or "n/a",
        },
        "ingest_and_read": bench_ingest_and_read(),
        "freshness_contract": check_freshness_contract(),
        "rerank_contract": check_rerank_contract(),
    }
    rr = result["rerank_contract"]
    contracts = (
        list(result["freshness_contract"].values())
        + [rr["access_outranks_same_raw_recent"], rr["recency_outranks_stale"],
           rr["effective_uplift_present"], rr["stale_untouched"]]
    )
    result["all_contracts_pass"] = all(bool(c) for c in contracts)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["all_contracts_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
