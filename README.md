<div align="center">

# FinKeyMemory

**Four-tier temporal memory for AI agents.**
Volatile working layer → as-of fact store → scene blocks → self-regenerating persona —
and a dream cycle that maintains all of them while your agent sleeps.

`pip install finkey-memory`

![License](https://img.shields.io/badge/license-AGPL--3.0-green)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-12%20suites-brightgreen)
![Deps](https://img.shields.io/badge/required%20deps-requests-lightgrey)

</div>

---

## The problem

Most agent memory is a vector database with a CRUD API on top. It answers "what did
the user say?" but never "what is true *now*, what *was* true, and how much do I
still care?" That's why agents confidently quote prices that changed three weeks
ago, and why their "user profile" is a pile of contradicting snippets.

FinKeyMemory treats memory as a **temporal, self-maintaining hierarchy**:

```
        ┌──────────────────────────────────────────────┐
  L3    │  Persona — regenerated from scene deltas     │  code-capped, JSON-validated
        ├──────────────────────────────────────────────┤
  L2    │  Scene blocks — grouped episodes + heat      │  scene summaries are embedded too
        ├──────────────────────────────────────────────┤
  L1    │  Facts with valid-from / as-of archive       │  superseded values are kept
        ├──────────────────────────────────────────────┤
  L0    │  Volatile working store (per process)        │  works with no infra at all
        └──────────────────────────────────────────────┘
     ⟲ Dream worker: decay, dedupe, pattern mining, L2/L3 regeneration
```

## What makes it different

- **Supersede ≠ delete.** When a fact changes, the old value is archived, not
  destroyed. The prompt block can say `(was: 92.5 until 2026-09-01)` — the model
  sees that the rate *moved*, which kills a whole class of stale-fact
  confidently-wrong answers.
- **Forgetting is a scoring law, not a calendar.** Ebbinghaus-style decay combined
  with *observed* access frequency: recall touches a fact → its heat grows; nobody
  needs it → it dims and the dream pass prunes or consolidates it.
- **The LLM never writes metadata.** Scene heat, summaries and persona bodies are
  produced under hard code caps and JSON validation. The model only proposes
  *assignments* (fact → scene) — it cannot silently rewrite its own memory state.
- **Self-regenerating persona.** A five-priority trigger ladder (explicit request /
  cold start / recovery / first run / threshold) rebuilds the user portrait from
  scene deltas, with a strict character budget that always fits your context window.
- **Degrades, never crashes.** No Postgres → in-process volatile store. No Qdrant →
  semantic layer off. No embedder → extraction skipped, existing rows untouched.
  The same code path works from a `python -c` experiment up to production.
- **PII masking on write** (phones/emails) and tenant-scoped keys
  (`company_id` / `user_id`) on every record.
- **Built-in eval harness** (`memory_eval.py`) with case/replay primitives, so you
  can measure recall before shipping — not vibes.

## The full feature map

Everything above plus these, all on by default:

| Feature | Module | What it gives you |
|---|---|---|
| Entity-hop recall | `entity_memory` | mentioning "Acme" or a person pulls multi-hop neighbours into context, no Neo4j |
| Cognition snapshots | `cognition_snapshot` | per-turn trace of perceived intent, core need, urgency — not just text |
| Episodic store | `episode` + `episode_codec` | events, not messages: who/what/when with stable serialization |
| Emotional traces + voice contract | `schema.UserImpression`, `has_voice_contract` | how the user reacts and how you must speak to them persist as first-class records |
| Conversation summaries | `summary_worker` | long chats compact themselves; raw transcripts don't eat your context |
| Background decay scheduler | `decay` | pruning/consolidation runs on a cadence, per tenant, safely from multiple processes |
| Dream cycle | `dream` | periodic prune + semantic dedupe + pattern mining; patterns persist as durable `pattern:<hash>` facts the next dream can reuse |
| Heat feedback from recall | `scene_persona` + `semantic` | a fact being *used* raises its heat — memory strengthens on access, dims on neglect |
| Parallel substrate loads | `manager.load_context` | ~12 remote round-trips run concurrently: memory adds single-digit ms to TTFT, not seconds |
| PII mask at write | `pii` | phones/emails never persist raw into vectors |
| Graceful degradation | everywhere | every missing backend disables exactly one layer; `python -c` demo and full production run the same code path |
| Tenant keys | every store | `(company_id, user_id)` on all records — multi-tenant is not a paid plugin |

## Quickstart

```bash
pip install "finkey-memory[pg,qdrant,redis]"   # or bare pip install finkey-memory for L0-only
```

```python
from finkey_memory import ExtractedFact
from finkey_memory.factory import build_memory_manager

# With no URLs/env set you get a zero-infra, in-process memory:
# facts are kept in the L0 volatile store and show up in load_context —
# nothing is silently dropped.
mm = build_memory_manager()

mm.upsert_extracted_facts(
    company_id="acme", user_id="u1", conversation_id="c1",
    facts=[ExtractedFact(key_normalized="favorite_city", value="Lisbon", source="user")],
)

ctx = mm.load_context("acme", "u1", "c1", "where should we travel?")
print(ctx.memory_prompt_block)   # property — dated, self-contained block for your system prompt
```

Production wiring is env-driven — point it at real backends and the same code
grows L1–L3 plus the background workers:

```bash
export DATABASE_URL=postgresql://user:pass@localhost/finkey_memory
export QDRANT_URL=http://localhost:6333
export REDIS_URL=redis://localhost:6379
export FINKEY_MEMORY_EXTRACTOR_BACKEND=openai    # or openrouter | ollama
export FINKEY_MEMORY_EXTRACTOR_MODEL=gpt-4o-mini
```

```python
from finkey_memory.factory import build_memory_manager
from finkey_memory.memory_eval import run_memory_eval   # measure recall, not vibes

mm = build_memory_manager()
mm.start_decay_scheduler()
mm.start_dream_worker()
```

Every feature is env-tunable (`FINKEY_MEMORY_SCENES`, `FINKEY_MEMORY_PERSONA`,
`FINKEY_MEMORY_TEMPORAL`, `FINKEY_MEMORY_PII_MASK`, `FINKEY_MEMORY_DECAY_ENABLED`, …)
and defaults to on.

## As-of, the killer demo

When two versions of the same fact land in the store, the older one is archived
(`{key}__asof__{ts}` soft keys, no schema migration) and the prompt block renders
the trajectory:

```python
mm.upsert_extracted_facts(company_id="acme", user_id="u1", conversation_id=None,
    facts=[ExtractedFact(key_normalized="mortgage_rate", value="92.5")])
# …a month later…
mm.upsert_extracted_facts(company_id="acme", user_id="u1", conversation_id=None,
    facts=[ExtractedFact(key_normalized="mortgage_rate", value="95.0")])

ctx = mm.load_context("acme", "u1", "c1", "what is the mortgage rate?")
# prompt block → mortgage_rate: 95.0 (was: 92.5 until 2026-09-01)
```

The model gets the *current* value **and** the movement — and says so naturally,
because the block is written to read like knowledge, not like a search dump.

## Benchmarks — reproducible, not vibes

`python bench/memory_bench.py` regenerates every number below on any machine
(pure Python, zero infra). Raw output is committed in
[`bench/RESULTS.md`](bench/RESULTS.md); competitor behaviour claims are only
made against their own source/docs, with file:line evidence, in
[`bench/SPEC_AUDIT.md`](bench/SPEC_AUDIT.md).

Measured on macOS arm64, Python 3.14 (10,000-fact tenant, L0 substrate):

| Metric | Value |
|---|---|
| ingest first 1,000 facts | **3.45 ms** |
| ingest 10,000 facts total | **35.6 ms** |
| `load_context` p50 / p95 / p99 | **4.0 / 4.3 / 5.8 ms** |
| semantic contracts (freshness, history visibility, tenant isolation, scoring law) | **7/7 pass, exit-code gated** |

Honest scope note: Postgres/Qdrant tiers and any head-to-head recall accuracy
against mem0/Zep/Letta are **not** published as numbers yet — the
LongMemEval-style harness in `bench/longmemeval/` is ready and we will commit
raw run outputs the moment datasets + keys are wired in. We would rather ship
an empty `results/` folder than a fabricated win.

## Architecture notes

| Layer | Storage | Authority |
|-------|---------|-----------|
| L0 volatile | in-process store | tenant-scoped facts before SQL exists |
| L1 facts | PostgreSQL `user_facts` + `__asof__` archive rows | code-owned validity windows |
| L2 scenes | PG columns + vector points (`deep_kind='scene'`) | LLM assigns, code computes heat |
| L3 persona | PG `user_persona` row, 2000-char hard cap | JSON-validated single response |
| ⟲ dream | background worker | prune / dedupe / mine / consolidate |

Backends are probed at startup and all optional; the manager exposes one
`load_context()` → `memory_prompt_block` path regardless of what's wired.
Fact extraction from raw conversations is pluggable: build an
`AsyncExtractorRunner` with your own recent-turns provider and attach it via
`mm.set_extractor_runner(...)` — or just upsert `ExtractedFact`s yourself.

## Development & tests

```bash
git clone https://github.com/NikitaKoreshkov/FinKeyMemory && cd FinKeyMemory
pip install -e ".[test]"
pytest            # 12 suites; pure-python by default, tests/live/ needs backends
```

13k LOC of production code across 38 modules, a 3k-line test suite, and no
required dependencies beyond `requests`.

## Contributing

PRs are welcome. By contributing you agree that copyright of your contribution is
assigned to FinKey, which lets us keep this project open under AGPL while also
offering commercial licences to companies that need them. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

AGPL-3.0-only. Use it, fork it, build SaaS on it — but if you modify it and serve
it over a network, your changes go back to the community. That's deliberate:
this is the memory engine of the FinKey agent platform, and cloud-copy-paste
protection is the price of publishing it. Commercial/licensing questions: open an
issue.
