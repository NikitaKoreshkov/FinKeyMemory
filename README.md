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

## What this is (and what it isn't)

**This is the memory layer, not a RAG pipeline.** It stores and recalls what is
true *about this user over time* (facts, scenes, persona) — it does not index your
documents. **RAG is optional and pluggable:** if you already have a knowledge-base
retriever, attach it with `mm.set_rag_engine(engine)` and its hits get merged into
`load_context(...)` alongside memory; with no engine set, memory runs completely
on its own. (The retrieval/RAG side lives in a separate FinKey component — this
package deliberately keeps them decoupled.)

**The model decides *what* to remember; code decides *how and where*.** After a few
turns, `MemoryExtractor` makes one cheap LLM call that reads the conversation and
proposes normalized facts — `{key, value, category, confidence, evidence}` — i.e. the
LLM chooses *which* details are worth keeping and how to phrase them. The code then
owns every storage decision the model must not be trusted with: dedup/conflict
resolution, supersession + as-of archive, PII masking, tenant scoping
(`company_id`/`user_id`), and the scoring law. The LLM never writes heat, validity
windows, or the persona directly (see below).

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

## Benchmarks — quality & correctness, not vibes

Regenerate everything with `python bench/suite.py` (deterministic, offline; the
parity figure reuses the committed strong-model run). Raw numbers land in
`bench/results/metrics.json`; competitor-behaviour claims are tied to their own
source file:line in [`bench/SPEC_AUDIT.md`](bench/SPEC_AUDIT.md).

| | |
|---|---|
| ![semantics](figures/semantics.png) | ![quality](figures/quality.png) |

**Correctness contracts: 100% (7/7, exit-code gated)** — freshness wins, supersession
history stays visible, tenant isolation, recency×access scoring law exact.

**Head-to-head vs mem0 (identical strong model `qwen3.8-max`, identical embedder, 5-conversation corpus):**

| Axis | FinKeyMemory | mem0 | Result |
|---|---|---|---|
| Fact-extraction quality | 19/19 (100%) | 19/19 (100%) | **parity** |
| Single authoritative current value | yes | no (keeps stale + new) | **ours** |
| No invented timestamps | yes | no (fabricates dates) | **ours** |

On a strong model we are **equal on extraction quality and ahead on temporal
semantics** — "your rate moved 92.5 → 95.0, current 95.0" is stored once, correctly,
without hallucinating when. We deliberately publish **no cross-system latency
number**: both pipelines call the same LLM per unit, so speed there is model-bound,
not a memory-layer differentiator. The scaling curve above is our *own* substrate
(offline, no LLM), shown for capacity planning, not as a competitive claim.

| Property | FinKeyMemory | typical mem0/Letta stack |
|---|---|---|
| Required dependencies | `requests` only | vector DB + embedding provider + (usually) LLM |
| Works with **no** Postgres/Qdrant/Redis/LLM | yes — L0 keeps serving, degrades silently | generally no |
| PII masking on write | built-in (phone/email) | not standard |
| Test suite | 12 suites / 3,011 LOC shipped | varies |
| License | AGPL-3.0 + contributor-assignment (commercial path) | varies (often Apache) |



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
