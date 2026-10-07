# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
FinKeyMemory benchmark suite — one command → metrics.json + figures/*.png.

    pip install "finkey-memory[bench]"
    python bench/suite.py

Generates:
  * figures/scaling.png    — OUR OWN substrate: ingest & read latency vs corpus
                             size (no competitor; speed-vs-mem0 is intentionally
                             not a headline because with a strong LLM both sides
                             are model-bound).
  * figures/quality.png    — extraction recall % vs mem0 on the IDENTICAL strong
                             model (from bench/results/qwen38max_parity.json).
  * figures/semantics.png  — temporal/correctness matrix vs mem0, scored from the
                             same measured run + SPEC_AUDIT file:line evidence.
  * metrics.json           — every number in one machine-readable file.

The LLM parity figure REUSES the committed run (no API calls here); to refresh it
run bench/compare_mem0_llm.py first. Everything else is deterministic and offline.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RESULTS = ROOT / "bench" / "results"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from finkey_memory import ExtractedFact
from finkey_memory.factory import build_memory_manager


def scaling_curve(sizes=(500, 1000, 2000, 5000, 10000)) -> dict:
    ingest, read_p50, read_p95 = [], [], []
    for n in sizes:
        mm = build_memory_manager()
        facts = [ExtractedFact(key_normalized=f"metric_{i % 200}_{i // 200}", value=f"value_{i}")
                 for i in range(n)]
        t0 = time.perf_counter()
        mm.upsert_extracted_facts(company_id="s", user_id="u", conversation_id=None, facts=facts)
        ingest.append((time.perf_counter() - t0) * 1000)
        lat = []
        for p in range(200):
            t1 = time.perf_counter()
            mm.load_context("s", "u", f"c{p % 5}", f"metric_{p % 200}")
            lat.append((time.perf_counter() - t1) * 1000)
        read_p50.append(statistics.median(lat))
        read_p95.append(sorted(lat)[int(len(lat) * 0.95)])
    return {"sizes": list(sizes), "ingest_total_ms": [round(x, 1) for x in ingest],
            "read_p50_ms": [round(x, 2) for x in read_p50],
            "read_p95_ms": [round(x, 2) for x in read_p95]}


def correctness_contracts() -> dict:
    checks = {}
    mm = build_memory_manager()
    mm.upsert_extracted_facts(company_id="k", user_id="u", conversation_id=None,
                              facts=[ExtractedFact(key_normalized="rate", value="92.5")])
    mm.upsert_extracted_facts(company_id="k", user_id="u", conversation_id=None,
                              facts=[ExtractedFact(key_normalized="rate", value="95.0")])
    blk = mm.load_context("k", "u", "c", "rate?").memory_prompt_block
    checks["single_authoritative"] = blk.count("rate:") == 1
    checks["current_is_new"] = "95.0" in blk
    checks["history_visible"] = "92.5" in blk
    other = mm.load_context("other", "u", "c", "rate?").memory_prompt_block
    checks["tenant_isolation"] = "rate:" not in other
    from finkey_memory.manager import _rerank_fact_hits
    now = time.time()
    ranked = _rerank_fact_hits([
        {"knowledge_key": "old", "score": 0.7, "created_at": now - 120 * 86400, "updated_at": now - 120 * 86400, "access_count": 0},
        {"knowledge_key": "hot", "score": 0.7, "created_at": now, "updated_at": now, "access_count": 40},
        {"knowledge_key": "rec", "score": 0.7, "created_at": now, "updated_at": now, "access_count": 0},
    ])
    order = [h["knowledge_key"] for h in ranked]
    checks["access_outranks_recent"] = order[0] == "hot"
    checks["recent_outranks_old"] = order.index("rec") < order.index("old")
    checks["stale_untouched"] = abs(ranked[[h["knowledge_key"] for h in ranked].index("old")]["score_effective"] - 0.7) < 1e-9
    passed = sum(1 for v in checks.values() if v)
    return {"checks": checks, "passed": passed, "total": len(checks),
            "pct": round(100.0 * passed / len(checks), 1)}


def load_parity() -> dict | None:
    p = RESULTS / "qwen38max_parity.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


# semantic matrix: 1 = property holds, 0 = documented/measured violation.
# finkey & mem0 scored on the SAME measured run (compare_mem0_llm) + our contracts;
# tencent from SPEC_AUDIT file:line. Evidence column keeps it honest.
SEMANTIC = [
    ("Single authoritative current value",        1, 0, 0, "measured run / l1-writer.ts:285 delete"),
    ("Superseded history stays visible",          1, 1, 0, "measured run / physical delete"),
    ("No invented timestamps",                     1, 0, None, "mem0 sample fabricated dates"),
    ("Dedup fails closed (not store-all)",         1, None, 0, "l1-dedup.ts:189 fail-open"),
    ("Serves reads with zero external services",   1, 0, 0, "L0 vs needs-vector-store"),
    ("PII masked on write",                        1, 0, None, "built-in vs not documented"),
    ("Ships test suite (3k LOC)",                  1, 1, 0, "0 test files in audited clone"),
]


def plot_scaling(sc: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = sc["sizes"]
    ax.plot(x, sc["ingest_total_ms"], marker="o", color="#5b8def", label="ingest total")
    ax.plot(x, sc["read_p95_ms"], marker="s", color="#e0a458", label="read p95")
    ax.plot(x, sc["read_p50_ms"], marker="^", color="#4caf7d", label="read p50")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("facts per tenant"); ax.set_ylabel("wall-clock (ms, log)")
    ax.set_title("FinKeyMemory substrate scaling (offline, deterministic)")
    ax.grid(True, which="both", alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(FIGS / "scaling.png", dpi=130); plt.close(fig)


def plot_quality(par: dict) -> None:
    users = list(par["mem0"]["needle_recall"].keys())
    f = [100 * par["finkeymemory"]["needle_recall"][u]["found"] / par["finkeymemory"]["needle_recall"][u]["of"] for u in users]
    m = [100 * par["mem0"]["needle_recall"][u]["found"] / par["mem0"]["needle_recall"][u]["of"] for u in users]
    xs = np.arange(len(users)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(xs - w/2, f, w, color="#4caf7d", label="FinKeyMemory")
    ax.bar(xs + w/2, m, w, color="#9a6bd6", label="mem0")
    ax.set_xticks(xs); ax.set_xticklabels(users); ax.set_ylim(0, 110)
    ax.set_ylabel("facts recalled (%)"); ax.set_title("Extraction quality — identical strong model (qwen3.8-max)")
    for i in range(len(users)):
        ax.text(i, 103, "parity" if f[i] == m[i] else "", ha="center", fontsize=8, color="#666")
    ax.legend(); fig.tight_layout(); fig.savefig(FIGS / "quality.png", dpi=130); plt.close(fig)


def plot_semantics() -> None:
    names = [s[0] for s in SEMANTIC]
    cols = ["FinKeyMemory", "mem0", "Tencent"]
    def cell(v):
        return ("\u2713", "#4caf7d") if v == 1 else ("\u2717", "#e05a5a") if v == 0 else ("n/a", "#b8b8b8")
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.set_xlim(0, len(cols)); ax.set_ylim(0, len(names))
    ax.set_xticks([i + 0.5 for i in range(len(cols))]); ax.set_xticklabels(cols, fontsize=11, weight="bold")
    ax.xaxis.tick_top()
    ax.set_yticks([i + 0.5 for i in range(len(names))]); ax.set_yticklabels(names, fontsize=10)
    for r, s in enumerate(SEMANTIC):
        y = r
        for c, val in enumerate((s[1], s[2], s[3])):
            sym, colr = cell(val)
            ax.add_patch(plt.Rectangle((c + 0.1, y + 0.12), 0.8, 0.76, color=colr, alpha=0.18))
            ax.text(c + 0.5, y + 0.5, sym, ha="center", va="center", fontsize=16, color=colr, weight="bold")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Temporal / correctness properties — measured run + source evidence", pad=28, fontsize=12)
    fig.tight_layout(); fig.savefig(FIGS / "semantics.png", dpi=130); plt.close(fig)


def main() -> None:
    sc = scaling_curve()
    cc = correctness_contracts()
    par = load_parity()
    metrics = {
        "substrate_scaling": sc,
        "correctness_contracts": cc,
        "semantic_matrix": [{"property": s[0], "finkey": s[1], "mem0": s[2],
                             "tencent": s[3], "evidence": s[4]} for s in SEMANTIC],
        "llm_parity": None if par is None else {
            "model": par["model"],
            "finkey_recall_rate": par["finkeymemory"]["needle_recall_rate"],
            "mem0_recall_rate": par["mem0"]["needle_recall_rate"],
            "finkey_asof": par["finkeymemory"]["asof"],
            "mem0_asof_stale_and_new": par["mem0"]["asof"]["returns_both_stale_and_new"],
        },
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_scaling(sc)
    plot_semantics()
    if par is not None:
        plot_quality(par)
    print(json.dumps({"correctness_pct": cc["pct"],
                      "correctness": f"{cc['passed']}/{cc['total']}",
                      "scaling_last": {"facts": sc["sizes"][-1],
                                        "ingest_ms": sc["ingest_total_ms"][-1],
                                        "read_p50_ms": sc["read_p50_ms"][-1]},
                      "parity": "present" if par else "run compare_mem0_llm.py",
                      "figures": sorted(p.name for p in FIGS.glob("*.png"))}, indent=2))


if __name__ == "__main__":
    main()
