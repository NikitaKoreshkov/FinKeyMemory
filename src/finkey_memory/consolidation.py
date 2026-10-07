# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Deterministic semantic-fact normalization, conflict handling, merging.

Pure Python — usable without embeddings. DB-agnostic canonical selection.
"""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from finkey_memory.schema import SemanticFactRecord, UserImpression


_WS = re.compile(r"\s+", re.UNICODE)
_NON_KEY = re.compile(r"[^\w\s\u0400-\u04FF]", re.UNICODE)


def normalize_fact_key(raw: str) -> str:
    """Stable key for upserts (Russian / Latin alphanumeric + spaces)."""
    s = (raw or "").strip().lower()
    s = _NON_KEY.sub("", s.replace("'", ""))
    s = _WS.sub(" ", s)
    out = s[:500].strip()
    return out if out else "unnamed_fact"


SOURCE_WEIGHT = {"user": 1.2, "verified": 1.15, "inferred": 1.0, "assistant": 0.9}


def _source_weight(src: Optional[str]) -> float:
    if not src:
        return SOURCE_WEIGHT["inferred"]
    return SOURCE_WEIGHT.get(str(src).lower(), 0.95)


def merge_fact_versions(
    existing_value: str,
    existing_confidence: float,
    existing_source: Optional[str],
    incoming_value: str,
    incoming_confidence: float,
    incoming_source: Optional[str],
) -> tuple[str, float, str]:
    """
    When the same normalized key receives a different value,
    prefer higher-trust sources, then confidence, then non-empty freshness.
    """
    ev_a = (_source_weight(existing_source), float(existing_confidence or 0.0), len(existing_value or ""))
    ev_b = (_source_weight(incoming_source), float(incoming_confidence or 0.0), len(incoming_value or ""))

    chosen_value: str
    chosen_confidence: float
    chosen_src: str

    if ev_b > ev_a:
        chosen_value, chosen_confidence, chosen_src = (
            incoming_value,
            incoming_confidence,
            incoming_source or "inferred",
        )
        if existing_value.strip() and existing_value.strip() != incoming_value.strip():
            alt = existing_value.strip()[:280]
            if alt not in chosen_value:
                chosen_value += f"\n(ранее: {alt})"
                chosen_confidence = min(0.99, chosen_confidence + 0.05)
    else:
        chosen_value, chosen_confidence, chosen_src = (
            existing_value,
            existing_confidence,
            existing_source or "inferred",
        )
        if incoming_value.strip() and incoming_value.strip() != existing_value.strip():
            alt = incoming_value.strip()[:280]
            if alt not in chosen_value:
                chosen_value += f"\n(alternative: {alt})"

    chosen_confidence = max(0.05, min(0.99, float(chosen_confidence)))
    return chosen_value.strip(), chosen_confidence, chosen_src


def clip_fact_value(text: str, max_chars: int = 1500) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1].rstrip() + "…"


def fact_rank_tuple(
    *,
    source: Optional[str],
    confidence: float,
    priority: float,
    touched_at: datetime,
) -> tuple[float, float, float, float, float]:
    """Descending sort key (higher wins). Tie-break by recency."""
    ts = touched_at.timestamp() if hasattr(touched_at, "timestamp") else 0.0
    return (_source_weight(source), float(priority), float(confidence), ts, ts)


def resolve_fact_records_group(records: list["SemanticFactRecord"]) -> Optional["SemanticFactRecord"]:
    """
    Pick canonical record from conflicting / duplicate rows sharing one logical fact.

    Rules: weighted source × priority × confidence × recency → take best spine,
    fold other values into the merged text via ``merge_fact_versions`` incrementally.
    """
    from finkey_memory.schema import SemanticFactRecord

    if not records:
        return None
    if len(records) == 1:
        return records[0]

    ordered = sorted(
        records,
        key=lambda r: fact_rank_tuple(
            source     = r.source,
            confidence = r.confidence,
            priority   = r.priority,
            touched_at = r.touched_at,
        ),
        reverse=True,
    )
    base = ordered[0]
    agg_val = base.value
    agg_conf = base.confidence
    agg_src = base.source or "inferred"
    agg_pri = base.priority

    touch = base.touched_at
    expiry_candidates = [e for e in (r.expires_at for r in records) if e is not None]
    merged_expiry = min(expiry_candidates, default=None) if expiry_candidates else None
    conf_candidates = [c for c in (r.confirmed_at for r in records) if c is not None]
    merged_confirmed = max(conf_candidates, default=base.confirmed_at)

    for other in ordered[1:]:
        touch = max(touch, other.touched_at)
        agg_pri = max(agg_pri, other.priority)
        agg_val, agg_conf, agg_src = merge_fact_versions(
            existing_value       = agg_val,
            existing_confidence = agg_conf,
            existing_source     = agg_src,
            incoming_value      = other.value,
            incoming_confidence = other.confidence,
            incoming_source     = other.source or "inferred",
        )

    return SemanticFactRecord(
        fact_key_normalized = ordered[0].fact_key_normalized,
        value               = clip_fact_value(agg_val),
        confidence          = max(0.05, min(0.99, float(agg_conf))),
        source              = agg_src or "inferred",
        priority            = max(0.0, min(1.0, float(agg_pri))),
        expires_at          = merged_expiry,
        confirmed_at        = merged_confirmed,
        created_at          = min(r.created_at for r in ordered),
        touched_at          = touch,
        id                  = ordered[0].id,
    )


def merge_user_impressions(a: "UserImpression", b: "UserImpression") -> "UserImpression":
    """Same ``impression_type`` channel: unify copy with conflict-aware merge."""
    from finkey_memory.schema import UserImpression

    rank_a = (_source_weight(a.source), float(a.confidence), float(a.priority))
    rank_b = (_source_weight(b.source), float(b.confidence), float(b.priority))
    prim, sec = (a, b) if rank_a >= rank_b else (b, a)
    v, c, s = merge_fact_versions(
        prim.content, prim.confidence, prim.source,
        sec.content, sec.confidence, sec.source,
    )
    pri = max(a.priority, b.priority)
    ea, eb = a.expires_at, b.expires_at
    expiry = min(ea, eb) if ea and eb else (ea or eb)
    lc_a, lc_b = a.last_confirmed_at, b.last_confirmed_at
    lc = max(lc_a, lc_b) if lc_a and lc_b else (lc_a or lc_b)
    conv = prim.source_conv_id or sec.source_conv_id

    return UserImpression(
        user_id          = prim.user_id,
        company_id       = prim.company_id,
        impression_type  = prim.impression_type,
        content          = clip_fact_value(v, max_chars=2800),
        confidence       = max(0.05, min(0.99, float(c))),
        source_conv_id   = conv,
        source           = (s or "inferred"),
        priority         = pri,
        expires_at       = expiry,
        last_confirmed_at= lc,
    )


def decay_stale_unconfirmed_facts(
    records: list["SemanticFactRecord"],
    *,
    now: datetime,
    stale_days: float = 90.0,
    decay_factor: float = 0.15,
) -> list["SemanticFactRecord"]:
    """
    Lower priority for old facts never confirmed — cheap forget without a graph walker.
    Returns new list (immutable-style replace).
    """
    from finkey_memory.schema import SemanticFactRecord as SFR

    out: list[SFR] = []
    cutoff = stale_days * 86400
    for r in records:
        age = now.timestamp() - r.touched_at.timestamp()
        if r.confirmed_at is None and age > cutoff:
            out.append(replace(r, priority=max(0.0, float(r.priority) - decay_factor)))
        elif r.expires_at is not None and now > r.expires_at and r.confirmed_at is None:
            out.append(replace(r, priority=max(0.0, float(r.priority) - 0.12)))
        else:
            out.append(r)
    return out
