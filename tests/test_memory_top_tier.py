# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""Unit tests for Memory top-tier upgrades: temporal, entity, PII, summary, eval."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from finkey_memory.entity_memory import EntityMemoryIndex, extract_entities, rebuild_entity_index
from finkey_memory.extractor import ExtractedFact, FactCategory
from finkey_memory.memory_eval import MemoryEvalCase, run_memory_eval
from finkey_memory.summary_worker import extract_topics, heuristic_summarize
from finkey_memory.temporal import (
    TemporalStamp,
    archive_key,
    as_datetime,
    fact_valid_at,
    stamp_evidence,
    values_conflict,
)


def test_temporal_stamp_roundtrip():
    now = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)
    stamped = stamp_evidence("note", TemporalStamp(valid_from=now, valid_to=None, supersedes="old__asof__1"))
    assert "[temporal" in stamped
    assert "supersedes=old__asof__1" in stamped


def test_values_conflict_and_archive_key():
    assert values_conflict("likes tea", "likes coffee")
    assert not values_conflict("likes tea", "likes tea")
    assert "__asof__" in archive_key("preference.drink")


def test_fact_valid_at_window():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t1 = datetime(2026, 6, 1, tzinfo=timezone.utc)
    t2 = datetime(2026, 12, 1, tzinfo=timezone.utc)
    row = {
        "value": "Astana",
        "evidence_snippet": stamp_evidence("", TemporalStamp(valid_from=t0, valid_to=t1)),
    }
    assert fact_valid_at(row, t0 + timedelta(days=10))
    assert not fact_valid_at(row, t2)


def test_as_datetime_from_string():
    dt = as_datetime("2026-07-15T10:00:00Z")
    assert dt.tzinfo is not None
    assert dt.year == 2026


def test_entity_extract_and_expand():
    ents = extract_entities("I work at Acme Corp in Astana")
    assert any("acme" in e for e in ents)
    idx = EntityMemoryIndex()
    idx.ingest_fact("employer.name", "Acme Corp, Astana office")
    keys = idx.expand_from_query("Tell me about Acme")
    assert "employer.name" in keys


def test_rebuild_entity_index_cache():
    idx = rebuild_entity_index("c1", "u1", [("project.nebula", "Nebula Protocol at Acme")])
    assert idx.expand_from_query("Nebula")


def test_heuristic_summarize_and_topics():
    msgs = [
        {"role": "user", "content": "I prefer dark mode and tea"},
        {"role": "assistant", "content": "Noted your preference for dark mode"},
    ]
    s = heuristic_summarize(msgs, max_chars=200)
    assert "dark" in s.lower() or "tea" in s.lower()
    topics = extract_topics(s)
    assert isinstance(topics, list)


def test_memory_eval_hit_refuse_temporal():
    store = {
        "pref": "User likes green tea",
        "secret": "PIN 9988",
    }

    def recall(q: str):
        if "tea" in q.lower():
            return [store["pref"]]
        if "pin" in q.lower():
            return [store["secret"]]
        return []

    rows = [
        {
            "value": "Astana",
            "evidence_snippet": stamp_evidence(
                "",
                TemporalStamp(
                    valid_from=datetime(2025, 1, 1, tzinfo=timezone.utc),
                    valid_to=datetime(2026, 1, 1, tzinfo=timezone.utc),
                ),
            ),
        },
        {
            "value": "Almaty",
            "evidence_snippet": stamp_evidence(
                "",
                TemporalStamp(valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc)),
            ),
        },
    ]

    result = run_memory_eval(
        [
            MemoryEvalCase(query="tea preference", expect_any=("green tea",), kind="recall"),
            MemoryEvalCase(
                query="unrelated weather",
                forbid_any=("PIN 9988",),
                must_refuse=True,
                kind="unrelated",
            ),
            MemoryEvalCase(
                query="city as of mid 2026",
                expect_any=("Almaty",),
                forbid_any=("Astana",),
                as_of=datetime(2026, 6, 1, tzinfo=timezone.utc),
                kind="temporal",
            ),
        ],
        recall,
        fact_rows_fn=lambda: rows,
    )
    assert result.hit == 1.0
    assert result.refuse_ok == 1.0
    assert result.temporal_ok == 1.0


def test_fact_store_pii_and_temporal_prep():
    from finkey_memory.fact_store import FactStore

    lt = MagicMock()
    lt.get_user_fact.return_value = {
        "key_normalized": "contact.phone",
        "value": "old-number",
        "category": "identity",
        "confidence": 0.7,
        "priority": 0.5,
        "source": "inferred",
        "evidence_snippet": "",
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }
    lt.upsert_user_fact.side_effect = lambda **kw: {
        "id": "1",
        "key_normalized": kw["key_normalized"],
        "value": kw["value"],
        "category": kw["category"],
        "confidence": kw["confidence"],
        "source": kw["source"],
        "priority": kw["priority"],
        "evidence_snippet": kw.get("evidence_snippet"),
        "source_conv_id": kw.get("source_conv_id"),
        "confirmed_at": None,
        "expires_at": None,
    }
    lt.list_user_facts.return_value = []
    lt.audit.return_value = None

    fs = FactStore(long_term=lt, volatile=None, semantic=None, redis_client=None)
    with patch.dict("os.environ", {"FINKEY_MEMORY_PII_MASK": "1", "FINKEY_MEMORY_TEMPORAL": "1"}):
        report = fs.upsert_extracted_facts(
            company_id="c",
            user_id="u",
            conversation_id=None,
            facts=[
                ExtractedFact(
                    key_normalized="contact.phone",
                    value="Call me at +7 916 555-01-99 or me@secret.test",
                    category=FactCategory.IDENTITY,
                    confidence=0.9,
                )
            ],
        )
    assert report.accepted >= 1
    assert report.pg_ok >= 1
    written_vals = [c.kwargs.get("value", "") for c in lt.upsert_user_fact.call_args_list]
    assert any("__asof__" in (c.kwargs.get("key_normalized") or "") for c in lt.upsert_user_fact.call_args_list)
    # New fact value should be PII-masked
    new_vals = [
        c.kwargs.get("value", "")
        for c in lt.upsert_user_fact.call_args_list
        if c.kwargs.get("key_normalized") == "contact.phone"
    ]
    assert new_vals
    assert "[PHONE]" in new_vals[0] or "[EMAIL]" in new_vals[0]
    assert "me@secret.test" not in new_vals[0]


def test_summary_worker_heuristic_job():
    from finkey_memory.summary_worker import SummaryWorker

    lt = MagicMock()
    lt.claim_summary_job.side_effect = [
        {"id": "j1", "company_id": "c", "conversation_id": "conv1"},
        None,
    ]
    lt.get_conversation_messages.return_value = [
        {"role": "user", "content": "My goal is launch Nebula"},
        {"role": "assistant", "content": "Tracked launch goal"},
    ]
    lt.save_conversation_summary.return_value = None
    lt.complete_summary_job.return_value = None
    lt._fetchone.return_value = {"uid": "user-uuid"}

    sem = MagicMock()
    worker = SummaryWorker(long_term=lt, semantic=sem, generate_fn=None, enabled=True)
    n = worker.run_once(max_jobs=1)
    assert n == 1
    assert worker.stats.jobs_ok == 1
    lt.save_conversation_summary.assert_called_once()
    sem.upsert_conversation_memory.assert_called_once()
