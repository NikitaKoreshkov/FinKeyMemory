# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Periodic consolidation / decay / vector refresh — call from cron or dequeue hook.

Keeps parity with summary workers: lightweight, deterministic, DB-agnostic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from finkey_memory.consolidation import decay_stale_unconfirmed_facts
from finkey_memory.volatile_store import consolidate_duplicate_normalization_pass, summarizefacts_for_embedding

if TYPE_CHECKING:
    from finkey_memory.semantic import SemanticMemory
    from finkey_memory.volatile_store import VolatileKnowledgeStore

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class MaintenanceReport:
    company_id:     str
    user_id:        str
    facts_in:       int
    facts_out:      int
    did_reindex_qdrant: bool
    errors:         list[str]


def run_user_memory_maintenance(
    *,
    store:           "VolatileKnowledgeStore",
    company_id:       str,
    user_id:          str,
    semantic: Optional["SemanticMemory"] = None,
    now:              Optional[datetime] = None,
) -> MaintenanceReport:
    """
    Steps:
      1) decay stale never-confirmed rows
      2) deterministic bucket-merge on normalized keys (defensive duplicate repair)
      3) rebuild one canonical JSON blob in Qdrant for ``deep_kind=semantic`` payloads
         (optional; safe to skip if Qdrant absent)
    """
    now = now or _now()
    errs: list[str] = []
    raw = store.all_fact_records(company_id, user_id)
    n_in = len(raw)

    decayed = decay_stale_unconfirmed_facts(raw, now=now)
    merged = consolidate_duplicate_normalization_pass(decayed)
    store.replace_facts(company_id, user_id, merged)
    n_out = len(merged)

    did_vec = False
    if semantic and merged:
        try:
            text = summarizefacts_for_embedding(company_id, merged)
            semantic.upsert_user_memory(
                company_id     = company_id,
                user_id        = user_id,
                memory_text    = text[:8000],
                memory_type    = "canonical_semantic_bundle",
                source_conv_id = None,
                deep_kind      = "semantic",
                payload_extra  = {"bundle": "maintenance_v1"},
            )
            did_vec = True
        except Exception as exc:
            errs.append(str(exc))
            logger.warning("Semantic reindex skipped: %s", exc)

    return MaintenanceReport(
        company_id         = company_id,
        user_id            = user_id,
        facts_in           = n_in,
        facts_out          = n_out,
        did_reindex_qdrant = did_vec,
        errors             = errs,
    )
