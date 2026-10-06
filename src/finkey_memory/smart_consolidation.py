# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Smart Memory Consolidation — LLM-guided semantic deduplication and importance scoring.

Extends the deterministic consolidation.py with:
  1. Cross-key semantic deduplication
     e.g. "user likes cats" and "user is a cat person" are the same fact
     Detected via embedding similarity, merged via LLM.

  2. Importance scoring
     Every fact gets a dynamic importance score based on:
       - Recency (Ebbinghaus forgetting curve)
       - Access frequency (how often retrieved)
       - Source credibility (user > verified > inferred > assistant)
       - Uniqueness (rare facts are more important than common ones)

  3. Cross-session pattern mining
     Identifies recurring themes / preferences across sessions
     → surfaces as long-term "personality insights" for the user.

  4. Memory garden (periodic maintenance)
     - Remove facts below importance threshold
     - Consolidate near-duplicate facts
     - Promote high-importance ephemeral to long-term
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional, Sequence

logger = logging.getLogger("finkey.memory.smart_consolidation")


_SOURCE_BASE = {"user": 1.0, "verified": 0.9, "inferred": 0.7, "assistant": 0.6}
_EBBINGHAUS_STABILITY = 7 * 24 * 3600


def ebbinghaus_retention(age_seconds: float, stability: float = _EBBINGHAUS_STABILITY) -> float:
    """R = e^(-t/S) — how much of a memory is retained after `age_seconds`."""
    return math.exp(-age_seconds / max(stability, 1.0))


@dataclass
class ImportanceScore:
    recency: float
    frequency: float
    source_credibility: float
    uniqueness: float
    total: float

    def to_dict(self) -> dict:
        return {
            "recency": round(self.recency, 3),
            "frequency": round(self.frequency, 3),
            "source_credibility": round(self.source_credibility, 3),
            "uniqueness": round(self.uniqueness, 3),
            "total": round(self.total, 3),
        }


def score_importance(
    *,
    value: str,
    source: Optional[str],
    touched_at: datetime,
    access_count: int = 0,
    max_access_count: int = 1,
    sibling_count: int = 0,
    now: Optional[datetime] = None,
    weights: Optional[dict] = None,
) -> ImportanceScore:
    """Compute a composite importance score for a memory fact."""
    w = weights or {"recency": 0.35, "frequency": 0.25, "source": 0.25, "uniqueness": 0.15}
    _now = now or datetime.now(timezone.utc)
    age = (_now - touched_at).total_seconds() if hasattr(touched_at, "timestamp") else 0.0

    recency = ebbinghaus_retention(max(0.0, age))
    frequency = min(1.0, (access_count or 0) / max(1, max_access_count))
    src_cred = _SOURCE_BASE.get(str(source or "").lower(), 0.6)
    uniqueness = 1.0 / (1.0 + max(0, sibling_count))

    total = (
        recency    * w["recency"]
        + frequency  * w["frequency"]
        + src_cred   * w["source"]
        + uniqueness * w["uniqueness"]
    )
    return ImportanceScore(
        recency=recency,
        frequency=frequency,
        source_credibility=src_cred,
        uniqueness=uniqueness,
        total=min(1.0, total),
    )



def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class DuplicateGroup:
    canonical_key: str
    members: list[str]
    similarity: float
    merged_value: Optional[str] = None


class SemanticDeduplicator:
    """
    Find semantically equivalent facts across different keys.

    e.g. "user_likes_cats" and "user_is_a_cat_person" → same fact.

    Parameters
    ----------
    embed_fn : callable(text) → list[float]
    merge_fn : callable(facts: list[str]) → str   (optional LLM merge)
    threshold : float   — cosine similarity threshold
    """

    def __init__(
        self,
        embed_fn: Callable[[str], list[float]],
        merge_fn: Optional[Callable] = None,
        threshold: float = 0.88,
    ) -> None:
        self._embed = embed_fn
        self._merge = merge_fn
        self._threshold = threshold

    def find_duplicates(
        self, facts: list[tuple[str, str]]
    ) -> list[DuplicateGroup]:
        """
        Find groups of semantically equivalent facts.
        Returns groups with ≥ 2 members.
        """
        if len(facts) < 2:
            return []

        embeddings = []
        for key, value in facts:
            try:
                emb = self._embed(value[:500])
                embeddings.append((key, value, emb))
            except Exception as exc:
                logger.warning("Embed failed for fact %s: %s", key, exc)

        visited = set()
        groups = []

        for i, (ki, vi, ei) in enumerate(embeddings):
            if ki in visited:
                continue
            group_members = [ki]
            for j, (kj, vj, ej) in enumerate(embeddings):
                if i == j or kj in visited:
                    continue
                sim = _cosine(ei, ej)
                if sim >= self._threshold:
                    group_members.append(kj)
                    visited.add(kj)

            if len(group_members) > 1:
                visited.add(ki)
                groups.append(DuplicateGroup(
                    canonical_key=ki,
                    members=group_members,
                    similarity=self._threshold,
                ))

        return groups

    def merge_group(self, group: DuplicateGroup, facts: dict[str, str]) -> str:
        """Merge all values in a duplicate group into one canonical value."""
        values = [facts.get(k, "") for k in group.members if facts.get(k)]
        if not values:
            return ""
        if len(values) == 1:
            return values[0]
        if self._merge:
            try:
                return self._merge(values)
            except Exception as exc:
                logger.warning("LLM merge failed: %s", exc)
        values.sort(key=len, reverse=True)
        return values[0]



def make_llm_merge_fn(generate_fn: Callable) -> Callable:
    """Create a merge function backed by an LLM."""
    def _merge(values: list[str]) -> str:
        joined = "\n".join(f"- {v}" for v in values[:10])
        prompt = (
            f"The following statements about the same fact/topic are near-duplicates. "
            f"Merge them into ONE concise, complete statement (≤100 words), "
            f"preserving all unique details:\n\n{joined}\n\nMerged:"
        )
        result = generate_fn([{"role": "user", "content": prompt}])
        return result.strip() if result else values[0]
    return _merge



@dataclass
class PatternInsight:
    pattern: str
    evidence_keys: list[str]
    confidence: float
    category: str

    def to_dict(self) -> dict:
        return {
            "pattern": self.pattern,
            "evidence": self.evidence_keys,
            "confidence": round(self.confidence, 3),
            "category": self.category,
        }


def mine_patterns_heuristic(
    facts: list[tuple[str, str]],
    min_occurrence: int = 3,
) -> list[PatternInsight]:
    """
    Detect recurring themes from memory keys/values without LLM.
    Fast, approximate, good enough for highlighting candidate insights.
    """
    import re
    from collections import Counter

    all_text = " ".join(v for _, v in facts).lower()
    words = re.findall(r"\b[а-яёa-z]{4,}\b", all_text)
    counter = Counter(words)

    stopwords = {
        "это", "что", "как", "его", "ее", "они", "мне", "что", "для",
        "than", "that", "this", "with", "have", "from", "they", "been",
    }
    insights = []
    for word, count in counter.most_common(20):
        if word in stopwords or count < min_occurrence:
            continue
        related = [k for k, v in facts if word in v.lower()]
        if len(related) >= min_occurrence:
            insights.append(PatternInsight(
                pattern=f"Recurring topic: '{word}'",
                evidence_keys=related[:5],
                confidence=min(0.9, count / len(facts)),
                category="topic",
            ))

    return insights[:10]



@dataclass
class GardenResult:
    kept: int
    pruned: int
    merged_groups: int
    pruned_keys: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"Garden: kept={self.kept}, pruned={self.pruned}, "
                f"merged_groups={self.merged_groups}")


def run_memory_garden(
    facts: list[tuple[str, str]],
    *,
    importance_scores: dict[str, ImportanceScore],
    duplicate_groups: list[DuplicateGroup],
    prune_threshold: float = 0.15,
) -> tuple[list[tuple[str, str]], GardenResult]:
    """
    Prune unimportant facts, merge duplicate groups.
    Returns (cleaned_facts, GardenResult).
    """
    keys_to_prune = {
        k for k, imp in importance_scores.items()
        if imp.total < prune_threshold
    }
    merged_keys = set()
    for group in duplicate_groups:
        for m in group.members[1:]:
            merged_keys.add(m)

    remove = keys_to_prune | merged_keys
    cleaned = [(k, v) for k, v in facts if k not in remove]

    return cleaned, GardenResult(
        kept=len(cleaned),
        pruned=len(facts) - len(cleaned),
        merged_groups=len(duplicate_groups),
        pruned_keys=list(remove),
    )
