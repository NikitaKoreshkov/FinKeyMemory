# Benchmark results — 2026-10-07

Every number below was produced by `python bench/memory_bench.py` on the
commit shown, and is reproducible with the same command (pure Python, no
network, no keys, no services). Exit code of the bench is 0 only when all
semantic contracts pass.

Machine: macOS 26.3 arm64, Python 3.14.3, Apple silicon.

## Substrate latency (L0 + prompt assembly, in-process)

| Measurement | Value |
|---|---|
| upsert first 1,000 facts | **3.45 ms** |
| upsert 10,000 facts total | **35.6 ms** |
| `load_context` p50 (10k-fact tenant, 200 queries) | **4.0 ms** |
| `load_context` p95 | **4.3 ms** |
| `load_context` p99 | **5.8 ms** |

Read latency is what gets added to your time-to-first-token when memory is
loaded per request — single-digit milliseconds with a 10k-fact tenant.

## Semantic contracts verified at runtime (7/7 pass)

- **current_value_in_block** — after a value change the block shows the new value.
- **superseded_value_visible** — the old value stays visible (`alternative/was:`
  annotation); supersede never destroys history.
- **tenant_isolation** — a second tenant sees nothing from the first.
- **access_outranks_same_raw_recent** — observed access adds on top of recency.
- **recency_outranks_stale** — inside the 90-day window, fresher wins at equal raw score.
- **effective_uplift_present** — `score_effective` differs from raw score for hot rows.
- **stale_untouched** — rows older than the window keep their raw score exactly
  (backwards-compatible with pre-scoring data).

Design note: bonuses are deliberately gentle (recency ≤ +15 %, access ≤ +5 %) —
retrieval score stays the primary signal; memory heat breaks ties, it does not
hijack ranking.

## What this bench does NOT measure

- Postgres/Qdrant/Redis tiers (as-of archive rendering, scene/persona cycles):
  covered by `tests/live/`, needs real services.
- Recall quality against conversation datasets: the LongMemEval/LoCoMo-style
  harness ships in `bench/longmemeval/` (runner + scoring; dataset download and
  LLM keys are yours by design). **We will not publish numbers we have not run** —
  no competitor head-to-head figures appear in this repo until that harness is
  executed end-to-end and the raw outputs are committed alongside.

## Comparison evidence

For competitor behaviour we publish only what is verifiable in their own source
or documentation — see `bench/SPEC_AUDIT.md`.
