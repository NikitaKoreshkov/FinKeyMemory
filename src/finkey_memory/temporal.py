# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Temporal fact validity — valid-from/valid-to windows on facts, no schema migration required.

When a fact value changes meaningfully, we stamp the outgoing value with a
``[temporal ...]`` marker in ``evidence_snippet`` before soft-delete/archive,
and stamp the new value with ``valid_from``.

Helpers reconstruct «what was true at time T» from active + archived rows
stored as ``{key}__asof__{unix}`` soft keys (or soft-deleted with temporal tag).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

_TEMPORAL_RE = re.compile(
    r"\[temporal\s+valid_from=(?P<vf>[^\s\]]+)"
    r"(?:\s+valid_to=(?P<vt>[^\s\]]+))?"
    r"(?:\s+supersedes=(?P<sup>[^\s\]]+))?\]",
    re.I,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _parse_iso(raw: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None


def as_datetime(value: object, *, fallback: Optional[datetime] = None) -> datetime:
    """Coerce PG/Redis timestamps to aware UTC datetime."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, str) and value.strip():
        parsed = _parse_iso(value.strip())
        if parsed is not None:
            return parsed
    return fallback or _now()


@dataclass(frozen=True)
class TemporalStamp:
    valid_from: datetime
    valid_to: Optional[datetime] = None
    supersedes: Optional[str] = None

    def to_tag(self) -> str:
        parts = [f"valid_from={_iso(self.valid_from)}"]
        if self.valid_to is not None:
            parts.append(f"valid_to={_iso(self.valid_to)}")
        if self.supersedes:
            parts.append(f"supersedes={self.supersedes}")
        return "[temporal " + " ".join(parts) + "]"


def parse_temporal_stamp(evidence: str) -> Optional[TemporalStamp]:
    m = _TEMPORAL_RE.search(evidence or "")
    if not m:
        return None
    vf = _parse_iso(m.group("vf") or "")
    if vf is None:
        return None
    vt = _parse_iso(m.group("vt")) if m.group("vt") else None
    return TemporalStamp(valid_from=vf, valid_to=vt, supersedes=m.group("sup"))


def stamp_evidence(evidence: str, stamp: TemporalStamp) -> str:
    base = _TEMPORAL_RE.sub("", evidence or "").strip()
    if base:
        return f"{stamp.to_tag()} {base}"[:2000]
    return stamp.to_tag()


def archive_key(key_normalized: str, when: Optional[datetime] = None) -> str:
    ts = int((when or _now()).timestamp())
    base = (key_normalized or "fact")[:60]
    return f"{base}__asof__{ts}"


def values_conflict(old: str, new: str) -> bool:
    a = re.sub(r"\s+", " ", (old or "").strip().lower())
    b = re.sub(r"\s+", " ", (new or "").strip().lower())
    if not a or not b:
        return False
    if a == b:
        return False
    # Same if one contains the other with small delta
    if a in b or b in a:
        return abs(len(a) - len(b)) > 24
    return True


def fact_valid_at(row: dict, as_of: datetime) -> bool:
    """True if this fact row is considered true at ``as_of``."""
    stamp = parse_temporal_stamp(str(row.get("evidence_snippet") or ""))
    created = row.get("created_at") or row.get("touched_at") or row.get("updated_at")
    if isinstance(created, str):
        created = _parse_iso(created)
    start = stamp.valid_from if stamp else created
    end = stamp.valid_to if stamp else None
    if start is not None and as_of < start:
        return False
    if end is not None and as_of >= end:
        return False
    # Soft-deleted without temporal end → not current
    if row.get("deleted_at") and stamp is None:
        return False
    if row.get("deleted_at") and stamp and stamp.valid_to is None:
        return False
    return True
