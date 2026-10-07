# Behavioural audit vs alternative agent-memory systems

Rule for this repo: a competitor claim is only allowed here if it is verifiable
in the competitor's **own published code** (file:line) or **official docs**.
No fabricated head-to-head numbers. Where we say "we win", the win is a tested
property of our code (`tests/`), matched against the competitor's documented
design.

## TencentDB-Agent-Memory (source audited locally, MIT, monorepo 199,840 TS LOC)

| Property | Tencent (evidence) | FinKeyMemory |
|---|---|---|
| Supersession of a changed fact | physical delete of old rows — `MemoryCore/src/core/record/l1-writer.ts:285` (`deleteL1Batch`) | archive + as-of annotation; history always readable (`bench/RESULTS.md`: `superseded_value_visible`) |
| Dedup failure mode | single batched LLM call; on error **fail-open to "store everything"** — `l1-dedup.ts:189` | conflict resolution in code (`consolidation.merge_fact_versions`), LLM proposes, code decides |
| Scene/persona content | raw markdown written by the LLM; deletion = the literal string `[DELETED]` in a file (`core/prompts/scene-extraction.ts:15,86`); META parsed from strings | LLM only assigns fact→scene and one summary line; heat/validity are PG columns mutated by code only; persona JSON-validated with hard code cap (`scene_persona.py`) |
| Scene recall | scenes were never embedded; navigation-only | scene summaries embedded (`deep_kind='scene'`), so recall itself feeds heat |
| Test coverage | 0 test files repo-wide (measured `*.test.ts`/`*.spec.ts` = 0 on the audited tree) | 12 suites / 3,011 test LOC; the semantic contracts above run in CI |
| Published benchmark | PersonaMem 48→76 % cited in README, but no eval code or dataset in the repo → not reproducible | we publish only benches reproducible from this repo with one command |

## mem0 (measured head-to-head + docs)

We ran mem0's production extraction path (`infer=True`) against FinKeyMemory's own
extractor on the **identical strong model** (`qwen/qwen3.8-max` via OpenRouter) and
the identical embedder, over a 5-conversation corpus with 19 ground-truth facts
(`bench/compare_mem0_llm.py`, raw output in `bench/results/`):

- **Extraction quality: parity** — both recovered 19/19 needles, both produced clean
  JSON. A strong LLM levels the recall; we do not claim to extract better.
- **Temporal supersession: ours** — after a value change, mem0 keeps the stale and
  the new value as two separate memories (`returns_both_stale_and_new = true`), so
  "what is the rate now" has no single authoritative answer in storage. FinKeyMemory
  stores one current value and archives the old (`single_authoritative = true`).
- **Date hallucination: theirs** — mem0's sample output invented dates that were not
  in the transcript ("as of October 7", "updated around the week of September 28").
  FinKeyMemory does not mint timestamps the conversation never stated.

Documented design points (their README/paper): LLM decides ADD/UPDATE/DELETE per
incoming fact, so delete-on-change is by design — the "what was true before" question
has no storage answer. (No latency claim: both sides call the same LLM per unit, so
speed is model-bound, not a memory-layer differentiator.)

## Letta (ex-MemGPT, docs)

Agent-managed memory blocks + archival memory: the agent rewrites its own core
memory via tools. Powerful, but the invariant "metadata is code-owned" is
inverted: the model mutates its own memory text. Our stance (`CONTRIBUTING.md`):
the LLM never writes heat, validity, or persona state directly. Different
trade-off, stated explicitly rather than hidden.

## Zep / Graphiti (docs)

Temporal knowledge graph with edge validity intervals — genuinely the closest
design to ours on the as-of axis. Difference: a graph database is a hard
dependency; FinKeyMemory's as-of works in flat Postgres rows with no migration
and a zero-infra L0 fallback. We consider them aligned in spirit and differ in
footprint; a measured head-to-head would require running their stack and is
reserved for the LongMemEval harness.

## What would change a "we win" into "we measured"

`bench/longmemeval/` — dataset, per-system adapters, identical prompts, both
raw outputs committed. Until then this document is the honest ceiling, and it
already shows the strongest case: delete-not-archive and fail-open dedup are
*their code*, and they are *our tested properties*.
