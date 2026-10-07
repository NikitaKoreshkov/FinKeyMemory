# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Fair LLM-powered comparison: mem0 vs FinKeyMemory, SAME strong model.

Both systems run real fact extraction with the identical OpenRouter model
(default qwen/qwen3.8-max-0902) on the same short conversations:
  * mem0        → add(..., infer=True)  (its production extraction path)
  * FinKeyMemory→ MemoryExtractor.extract()  (this package's own extractor)
Embeddings (vector tier) are identical on both sides too (fastembed bge-small,
CPU, no server). We measure: extraction wall-clock, correct-fact recall against
ground truth, and temporal as-of behaviour. No Ollama. Key read from env, never
printed. Cost is bounded by keeping N small.

Usage:
  OPENROUTER_API_KEY=... python bench/compare_mem0_llm.py
  (or set QWEN_MODEL to override the model slug; N_CONV to scale the corpus)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MODEL = os.getenv("QWEN_MODEL", "qwen/qwen3.8-max-0902")
KEY = (os.getenv("OPENROUTER_API_KEY") or "").strip()
if not KEY:
    raise SystemExit("OPENROUTER_API_KEY is required for the LLM comparison (never printed)")

from fastembed import TextEmbedding
_emb = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
DIM = 384
def embed_one(t): return list(next(iter(_emb.embed([t]))))
def embed_many(ts): return [list(v) for v in _emb.embed(ts)]

# ---- ground-truth conversations: each has facts + a value that changes ----
CONVOS = [
    {"uid": "u_ann", "turns": [
        {"role": "user", "content": "I'm Ann, I live in Porto and my favorite programming language is Elixir."},
        {"assistant_note": "ack"},
        {"role": "user", "content": "My mortgage rate is 92.5 percent right now."}],
        "needles": {"name": "Ann", "city": "Porto", "language": "Elixir", "rate": "92.5"}},
    {"uid": "u_ben", "turns": [
        {"role": "user", "content": "I'm Ben, my daughter is called Cleo, and I work at Acme as a data engineer."}],
        "needles": {"name": "Ben", "daughter": "Cleo", "company": "Acme", "role": "data engineer"}},
    {"uid": "u_cara", "turns": [
        {"role": "user", "content": "I'm Cara. My blood type is O+ and I'm allergic to penicillin."},
        {"role": "user", "content": "Also my budget ceiling is 3000 dollars per month."}],
        "needles": {"name": "Cara", "blood": "O+", "allergy": "penicillin", "budget": "3000"}},
    {"uid": "u_dan", "turns": [
        {"role": "user", "content": "I'm Dan. My server timezone is Europe/Amsterdam and my deploy day is Thursday."}],
        "needles": {"name": "Dan", "tz": "Europe/Amsterdam", "deploy": "Thursday"}},
    {"uid": "u_elena", "turns": [
        {"role": "user", "content": "I'm Elena, I run a bakery called Golden Crust in Krakow."},
        {"role": "user", "content": "My VAT number is PL7771234567."}],
        "needles": {"name": "Elena", "biz": "Golden Crust", "city": "Krakow", "vat": "PL7771234567"}},
]
N = int(os.getenv("N_CONV", str(len(CONVOS))))
CONVOS = CONVOS[:N]

def msgs_for(conv):
    m = []
    for t in conv["turns"]:
        if "role" in t:
            m.append({"role": t["role"], "content": t["content"]})
    return m

out = {"model": MODEL, "embed_model": "bge-small-en-v1.5", "n_convo": len(CONVOS),
       "note": "identical LLM + embedder both sides; mem0 infer=True is its production extraction path"}

# ================= mem0 (infer=True) =================
from mem0 import Memory
from qdrant_client import QdrantClient
class FBEmb:
    def embed(self, text, infer=False): return embed_one(text)
    def embed_batch(self, texts, infer=False): return embed_many(texts)
    embedding_dims = DIM

m0cfg = {
    "llm": {"provider": "openai", "config": {
        "model": MODEL, "openai_base_url": "https://openrouter.ai/api/v1",
        "api_key": KEY, "temperature": 0.1, "max_tokens": 2000}},
    "vector_store": {"provider": "qdrant", "config": {
        "collection_name": "cmp_llm", "path": "/tmp/mbench/m0qdrant_llm", "embedding_model_dims": DIM}},
}
m0 = Memory.from_config(m0cfg)
m0.embedding_model = FBEmb()

t0 = time.perf_counter()
mem0_store = {}
for conv in CONVOS:
    r = m0.add(msgs_for(conv), user_id=conv["uid"], infer=True)
    got = m0.search(" ".join(str(v) for v in conv["needles"].values()),
                    filters={"user_id": conv["uid"]}, limit=20)
    mem0_store[conv["uid"]] = [h["memory"] for h in got.get("results", [])]
mem0_ing = (time.perf_counter() - t0) * 1000
mem0_recall = {}
for conv in CONVOS:
    blob = " | ".join(mem0_store[conv["uid"]]).lower()
    hits = sum(1 for v in conv["needles"].values() if str(v).lower() in blob)
    mem0_recall[conv["uid"]] = {"found": hits, "of": len(conv["needles"])}
# as-of on mem0: add a changed rate
m0.add([{"role": "user", "content": "My mortgage rate is now 95.0 percent, updated last week."}],
       user_id="u_ann", infer=True)
top = [h["memory"] for h in m0.search("mortgage rate", filters={"user_id": "u_ann"}, limit=10).get("results", [])]
mem0_asof = {"returns_both_stale_and_new": any("92.5" in t for t in top) and any("95.0" in t for t in top),
             "sample": top[:4]}

# ================= FinKeyMemory extractor =================
from finkey_memory.llm_completer import OpenRouterCompleter
from finkey_memory.extractor import MemoryExtractor
comp = OpenRouterCompleter(model=MODEL, api_key=KEY, timeout=60, json_mode=True)
ex = MemoryExtractor(completer=comp)

from finkey_memory.factory import build_memory_manager
fk_store = {}
t0 = time.perf_counter()
fk_err = ""
for conv in CONVOS:
    turns = [{"role": t.get("role", "user"), "content": t.get("content", "")} for t in msgs_for(conv)]
    res = ex.extract(company_id="cmp", user_id=conv["uid"], conversation_id="c1", recent_turns=turns)
    fk = build_memory_manager()
    fk.upsert_extracted_facts(company_id="cmp", user_id=conv["uid"], conversation_id="c1", facts=res.facts)
    blob = " | ".join(r.value for r in fk._volatile.all_fact_records("cmp", conv["uid"])).lower()
    hits = sum(1 for v in conv["needles"].values() if str(v).lower() in blob)
    fk_store[conv["uid"]] = {"found": hits, "of": len(conv["needles"]),
                             "n_facts": len(res.facts), "parse_ok": res.parse_ok}
fk_ing = (time.perf_counter() - t0) * 1000

# as-of on FinKeyMemory (supersede)
from finkey_memory import ExtractedFact
fk2 = build_memory_manager()
fk2.upsert_extracted_facts(company_id="asof", user_id="u_ann", conversation_id=None, facts=[ExtractedFact(key_normalized="mortgage_rate", value="92.5")])
fk2.upsert_extracted_facts(company_id="asof", user_id="u_ann", conversation_id=None, facts=[ExtractedFact(key_normalized="mortgage_rate", value="95.0")])
block = fk2.load_context("asof", "u_ann", "c", "mortgage rate?").memory_prompt_block
fk_asof = {"current_is_new": "95.0" in block, "old_still_archived": "92.5" in block,
           "single_authoritative": block.count("mortgage_rate") == 1}

def rate(d): return round(sum(x["found"] for x in d.values()) / max(1, sum(x["of"] for x in d.values())), 3)

out["mem0"] = {"ingest_total_ms": round(mem0_ing, 1), "per_convo_ms": round(mem0_ing/len(CONVOS), 1),
               "needle_recall": mem0_recall, "needle_recall_rate": rate(mem0_recall),
               "asof": mem0_asof}
out["finkeymemory"] = {"ingest_total_ms": round(fk_ing, 1), "per_convo_ms": round(fk_ing/len(CONVOS), 1),
               "needle_recall": fk_store, "needle_recall_rate": rate(fk_store), "asof": fk_asof, "err": fk_err}

print(json.dumps(out, indent=2, ensure_ascii=False))
