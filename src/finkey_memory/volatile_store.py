# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Working-set memory until your persistence layer exists.

Facts and impressions consolidate on write using ``memory.consolidation``.
Episodes / traces / procedural strings are capped per user to stay RAM-bounded.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from threading import Lock
from typing import Callable, Iterable, Iterator, Tuple

from finkey_memory.consolidation import (
    clip_fact_value,
    merge_user_impressions,
    normalize_fact_key,
    resolve_fact_records_group,
)
from finkey_memory.schema import SemanticFactRecord, UserImpression


TenantKey = Tuple[str, str]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class VolatileKnowledgeStore:
    """
    Tenant-scoped in-process store.
    Suitable for gateways without DATABASE_URL until you persist via adapter.
    """

    def __init__(
        self,
        max_episode_snapshots: int = 400,
        max_emotional_traces: int = 500,
    ) -> None:
        self._lock = Lock()
        self._max_epi = max_episode_snapshots
        self._max_emo = max_emotional_traces

        self._facts: dict[TenantKey, dict[str, SemanticFactRecord]] = defaultdict(dict)
        self._eps: dict[TenantKey, list[dict]] = defaultdict(list)
        self._emo: dict[TenantKey, list[str]] = defaultdict(list)
        self._proc: dict[TenantKey, dict[str, tuple[str, list[str], int]]] = defaultdict(dict)
        self._imps: dict[TenantKey, dict[str, UserImpression]] = defaultdict(dict)

        self._interaction_count: dict[TenantKey, int] = defaultdict(int)

    def tenant_keys(self) -> Iterator[TenantKey]:
        with self._lock:
            tenants = set(self._facts.keys()) | set(self._eps.keys()) | set(self._imps.keys())
        for t in tenants:
            yield t

    def bump_interaction(self, company_id: str, user_id: str) -> int:
        tk: TenantKey = (company_id, user_id)
        with self._lock:
            self._interaction_count[tk] += 1
            return self._interaction_count[tk]

    def upsert_fact(
        self,
        company_id:   str,
        user_id:      str,
        raw_key:      str,
        value:        str,
        *,
        confidence:   float                       = 0.55,
        source:       str                         = "inferred",
        priority:    float                       = 0.5,
        expires_at:   datetime | None             = None,
        confirmed_at: datetime | None             = None,
    ) -> None:
        nk = normalize_fact_key(raw_key)
        val = clip_fact_value(value)
        now = _now()

        incomer = SemanticFactRecord(
            fact_key_normalized = nk,
            value               = val,
            confidence          = confidence,
            source              = source,
            priority            = max(0.0, min(1.0, priority)),
            expires_at          = expires_at,
            confirmed_at        = confirmed_at,
            created_at          = now,
            touched_at          = now,
        )

        tk: TenantKey = (company_id, user_id)
        with self._lock:
            bucket = self._facts[tk]
            if nk in bucket:
                merged = resolve_fact_records_group([bucket[nk], incomer])
                if merged:
                    merged = replace(merged, touched_at=_now())
                    bucket[nk] = merged
            else:
                bucket[nk] = incomer

    def delete_fact(self, company_id: str, user_id: str, raw_key: str) -> None:
        """Удалить факт из in-process кэша (после soft-delete в PG)."""
        nk = normalize_fact_key(raw_key)
        tk: TenantKey = (company_id, user_id)
        with self._lock:
            self._facts.get(tk, {}).pop(nk, None)

    def clear_facts(self, company_id: str, user_id: str) -> None:
        """Очистить все semantic-факты тенанта в volatile-кэше."""
        tk: TenantKey = (company_id, user_id)
        with self._lock:
            self._facts.pop(tk, None)

    def all_fact_records(self, company_id: str, user_id: str) -> list[SemanticFactRecord]:
        tk: TenantKey = (company_id, user_id)
        with self._lock:
            return list(self._facts.get(tk, {}).values())

    def replace_facts(self, company_id: str, user_id: str, records: list[SemanticFactRecord]) -> None:
        tk: TenantKey = (company_id, user_id)
        with self._lock:
            self._facts[tk] = {r.fact_key_normalized: r for r in records}

    def fact_lines_for_prompt(
        self,
        company_id: str,
        user_id: str,
        *,
        now: datetime | None = None,
        limit: int            = 20,
        min_priority: float   = 0.08,
    ) -> list[str]:
        now = now or _now()
        recs: list[SemanticFactRecord] = []
        for r in self.all_fact_records(company_id, user_id):
            if r.priority < min_priority:
                continue
            if r.expires_at is not None and now > r.expires_at:
                continue
            recs.append(r)
        recs.sort(key=lambda x: (-x.priority, -x.confidence, -x.touched_at.timestamp()))
        return [f"{r.fact_key_normalized}: {r.value}" for r in recs[:limit]]

    def save_memory_episode_snapshot(
        self,
        company_id: str,
        user_id: str,
        snapshot_dict: dict,
        episode_client_id: str,
    ) -> None:
        tk: TenantKey = (company_id, user_id)
        row = dict(snapshot_dict)
        row["_episode_client_id"] = episode_client_id
        row["_saved_at"] = _now().isoformat()

        with self._lock:
            buf = self._eps[tk]
            buf.insert(0, row)
            if len(buf) > self._max_epi:
                del buf[self._max_epi :]

    def recent_episode_lines(self, company_id: str, user_id: str, limit: int = 6) -> list[str]:
        tk: TenantKey = (company_id, user_id)
        with self._lock:
            rows = list(self._eps.get(tk, [])[:limit])
        lines: list[str] = []
        for snap in rows:
            topic = str(snap.get("topic") or "")[:120]
            emo = str(snap.get("user_emotion") or "")
            need = str(snap.get("core_need_read") or "")[:200]
            frag = " | ".join(x for x in (topic, emo, need) if x)
            if frag:
                lines.append(frag)
        return lines

    def append_emotional_trace(
        self,
        company_id: str,
        user_id: str,
        *,
        user_emotion: str,
        user_intensity: float,
        finkey_tenderness: float,
        resonance: float,
        finkey_inner_feeling: str,
    ) -> None:
        line = (
            f"{user_emotion} (i={float(user_intensity):.2f}, "
            f"тендерность={float(finkey_tenderness):.2f}, r={float(resonance):.2f}): "
            f"{finkey_inner_feeling[:240]}"
        ).strip()

        tk: TenantKey = (company_id, user_id)
        with self._lock:
            lst = self._emo[tk]
            lst.insert(0, line)
            if len(lst) > self._max_emo:
                del lst[self._max_emo :]

    def recent_emotional_lines(self, company_id: str, user_id: str, limit: int = 7) -> list[str]:
        tk: TenantKey = (company_id, user_id)
        with self._lock:
            return list(self._emo.get(tk, [])[:limit])

    def upsert_procedural_memory(
        self,
        company_id: str,
        user_id: str,
        title: str,
        content: str,
        tags: list[str] | None = None,
    ) -> None:
        tkey = (title or "note").strip()[:500] or "note"
        tk: TenantKey = (company_id, user_id)

        with self._lock:
            prev = self._proc[tk].get(tkey)
            use = prev[2] + 1 if prev else 1
            self._proc[tk][tkey] = (clip_fact_value(content, max_chars=4000), tags or [], use)

    def procedural_lines(self, company_id: str, user_id: str, limit: int = 6) -> list[str]:
        tk: TenantKey = (company_id, user_id)
        with self._lock:
            items = sorted(
                self._proc.get(tk, {}).items(),
                key=lambda kv: kv[1][2],
                reverse=True,
            )[:limit]
        return [f"{k}: {v[0]}" for k, v in items]

    def upsert_impression(self, impression: UserImpression) -> None:
        cid, uid, itype = impression.company_id, impression.user_id, impression.impression_type
        tk: TenantKey = (cid, uid)

        with self._lock:
            bucket = self._imps[tk]
            if itype not in bucket:
                bucket[itype] = impression
            else:
                bucket[itype] = merge_user_impressions(bucket[itype], impression)

    def list_impressions(self, company_id: str, user_id: str, limit: int = 48) -> list[UserImpression]:
        tk: TenantKey = (company_id, user_id)
        with self._lock:
            rows = sorted(
                self._imps.get(tk, {}).values(),
                key=lambda x: (-x.priority, -x.confidence),
            )[:limit]
        return list(rows)


def summarizefacts_for_embedding(company_id: str, records: Iterable[SemanticFactRecord]) -> str:
    payload = [{"k": r.fact_key_normalized, "v": r.value, "pri": round(r.priority, 3)} for r in records]
    blob = {"company_id": company_id, "canonical_facts": payload}
    return json.dumps(blob, ensure_ascii=False)


def consolidate_duplicate_normalization_pass(
    records: list[SemanticFactRecord],
    *,
    key_normalizer: Callable[[str], str] = normalize_fact_key,
) -> list[SemanticFactRecord]:
    """
    Offline pass: buckets by ``key_normalizer`` (usually already normalized).

    Keeps deterministic canonical row per bucket.
    """
    groups: dict[str, list[SemanticFactRecord]] = defaultdict(list)
    for r in records:
        groups[key_normalizer(r.fact_key_normalized)].append(r)
    out: list[SemanticFactRecord] = []
    for _k, group in groups.items():
        merged = resolve_fact_records_group(group)
        if merged:
            out.append(merged)
    return sorted(out, key=lambda r: (-r.priority, r.fact_key_normalized))
