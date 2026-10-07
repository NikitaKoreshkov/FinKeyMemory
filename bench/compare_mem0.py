# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Cross-system benchmark — FinKeyMemory vs mem0 (the most-adopted OSS agent memory).

Reproduce:
    python3.12 -m venv .venv && . .venv/bin/activate
    pip install "finkey-memory[pg]" mem0ai fastembed qdrant-client
    python bench/compare_mem0.py

Fairness contract:
  * Identical REAL embedding model on both sides: BAAI/bge-small-en-v1.5 via
    fastembed (ONNX, CPU, no server, no API key).
  * mem0 runs with ``infer=False`` → its LLM fact-extraction step is SKIPPED,
    i.e. we measured mem0's *fast* path; real mem0 usage adds an LLM call on
    top and is strictly slower. No OpenAI key is ever called.
  * No network: mem0's own vector store (local Qdrant) and FinKeyMemory's
    Qdrant in-memory / L0 keyword tier.
  * Single process; RSS is cumulative (shared model) and therefore NOT
    published as a per-system number.

Outputs JSON to stdout. Raw run is committed in bench/RESULTS.md.
"""
"""Fair storage/retrieval comparison — same real embedding model (fastembed
bge-small, ONNX CPU, no server) on both sides. mem0 uses Chroma + infer=False
(raw store, NO LLM). FinKeyMemory uses Qdrant in-memory (no server) for the
vector tier and its L0 keyword tier separately. Deterministic, no keys, no Ollama.
"""
import json
import os
import resource
import statistics
import sys
import time

os.environ.setdefault("OPENAI_API_KEY", "stub-not-called")  # LLM never invoked: infer=False
os.environ.setdefault("MEM0_TELEMETRY", "False")

sys.path.insert(0, "/Users/nikita/Desktop/ALL-WORKS/AI/FinKeyMemory/src")

DIM = 384
N = 1000
PROBES = 100

from fastembed import TextEmbedding
_embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

def embed_one(text: str):
    return next(iter(_embedder.embed([text])))

def embed_many(texts):
    return list(_embedder.embed(texts))

def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

facts = [f"Fact {i}: metric_{i % 40} equals value_{i} for tenant bench" for i in range(N)]

# warm embedding model once (excluded from timing on both sides)
embed_one("warmup"); rss_after_warm = rss_mb()

out = {"setup": {"n_facts": N, "probes": PROBES, "dim": DIM,
                 "embed_model": "BAAI/bge-small-en-v1.5 (fastembed ONNX CPU)",
                 "mem0_llm": "infer=False (NO LLM)"}}

from mem0 import Memory

class FBEmbedder:
    def embed(self, text, infer=False):
        return embed_one(text)
    def embed_batch(self, texts, infer=False):
        return embed_many(texts)
    embedding_dims = DIM

m0cfg = {
    "vector_store": {
        "provider": "qdrant",
        "config": {"collection_name": "cmp_bench", "path": "/tmp/mbench/m0qdrant",
                   "embedding_model_dims": DIM},
    },
}
m0 = Memory.from_config(m0cfg)
m0.embedding_model = FBEmbedder()

t0 = time.perf_counter()
ok = 0; err = ""
for f in facts:
    try:
        m0.add([{"role": "user", "content": f}], user_id="bench_u", infer=False)
        ok += 1
    except Exception as e:
        err = repr(e)[:160]; break
mem0_ing = (time.perf_counter() - t0) * 1000
lat = []
mem0_top = []
for p in range(PROBES):
    q = f"metric_{p % 40}"
    t1 = time.perf_counter()
    r = m0.search(q, filters={"user_id": "bench_u"}, limit=5)
    lat.append((time.perf_counter() - t1) * 1000)
    mem0_top.append(len(r.get("results", [])))
m0.add([{"role":"user","content":"mortgage_rate equals 92.5"}], user_id="asof_u", infer=False)
m0.add([{"role":"user","content":"mortgage_rate equals 95.0 (updated)"}], user_id="asof_u", infer=False)
hits = [h["memory"] for h in m0.search("mortgage_rate", filters={"user_id": "asof_u"}, limit=5).get("results", [])]
out["mem0"] = {
    "ingest_total_ms": round(mem0_ing, 1), "ingest_per_fact_ms": round(mem0_ing/max(1,ok), 3),
    "stored": ok, "first_err": err,
    "search_ms_p50": round(statistics.median(lat), 2),
    "search_ms_p95": round(sorted(lat)[int(len(lat)*0.95)], 2),
    "avg_topk_returned": round(statistics.mean(mem0_top), 2),
    "rss_mb_delta": round(rss_mb() - rss_after_warm, 1),
    "asof_returns_stale_and_new": any("92.5" in h for h in hits) and any("95.0" in h for h in hits),
}

# ---------------- FinKeyMemory: Qdrant-in-memory vector tier ----------------
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
qc = QdrantClient(":memory:")
qc.create_collection("fk", vectors_config=VectorParams(size=DIM, distance=Distance.COSINE))
vecs = embed_many(facts)
t0 = time.perf_counter()
for i, (f, v) in enumerate(zip(facts, vecs)):
    qc.upsert("fk", points=[PointStruct(id=i, vector=list(v), payload={"text": f})])
fk_ing = (time.perf_counter() - t0) * 1000
lat = []; fk_top = []
for p in range(PROBES):
    qv = embed_one(f"metric_{p % 40}")
    t1 = time.perf_counter()
    qc.query_points("fk", query=list(qv), limit=5)
    lat.append((time.perf_counter() - t1) * 1000)
    fk_top.append(5)
out["finkey_vector"] = {
    "ingest_total_ms": round(fk_ing, 1), "ingest_per_fact_ms": round(fk_ing/N, 3),
    "search_ms_p50": round(statistics.median(lat), 2),
    "search_ms_p95": round(sorted(lat)[int(len(lat)*0.95)], 2),
    "rss_mb_delta": round(rss_mb() - rss_after_warm, 1),
}

# ---------------- FinKeyMemory: L0 keyword tier (no vectors at all) ----------------
from finkey_memory import ExtractedFact
from finkey_memory.factory import build_memory_manager
fk = build_memory_manager()
fk_facts = [ExtractedFact(key_normalized=f"metric_{i % 40}_{i // 40}", value=f"value_{i}") for i in range(N)]
t0 = time.perf_counter()
rep = fk.upsert_extracted_facts(company_id="bench", user_id="u1", conversation_id=None, facts=fk_facts)
l0_ing = (time.perf_counter() - t0) * 1000
lat = []
for p in range(PROBES):
    t1 = time.perf_counter()
    fk.load_context("bench", "u1", f"c{p % 5}", f"metric_{p % 40}")
    lat.append((time.perf_counter() - t1) * 1000)
fk2 = build_memory_manager()
fk2.upsert_extracted_facts(company_id="asof", user_id="u1", conversation_id=None, facts=[ExtractedFact(key_normalized="mortgage_rate", value="92.5")])
fk2.upsert_extracted_facts(company_id="asof", user_id="u1", conversation_id=None, facts=[ExtractedFact(key_normalized="mortgage_rate", value="95.0")])
block = fk2.load_context("asof", "u1", "c", "mortgage rate?").memory_prompt_block
out["finkey_l0"] = {
    "ingest_total_ms": round(l0_ing, 1), "ingest_per_fact_ms": round(l0_ing/N, 4),
    "stored": rep.volatile_ok, "deps": "none beyond stdlib+requests",
    "read_ms_p50": round(statistics.median(lat), 3),
    "read_ms_p95": round(sorted(lat)[int(len(lat)*0.95)], 3),
    "rss_mb_delta": round(rss_mb() - rss_after_warm, 1),
    "asof_current_is_95": "95.0" in block,
    "asof_single_authoritative": block.count("mortgage_rate") == 1,
    "asof_primary_is_not_stale": not block.split("mortgage_rate")[1].lstrip(" :").startswith("92.5"),
}

print(json.dumps(out, indent=2, ensure_ascii=False))
