# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Entity memory graph — entity-hop recall without a Neo4j dependency.

Builds an in-process + optional Redis adjacency map:
  entity_normalized → {fact_keys, related_entities}

Used at recall time to expand prompt context with multi-hop neighbours
when the user mentions an entity ("Acme", "Астана", person names).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

_ENTITY_RE = re.compile(
    r"\b([A-ZА-ЯЁ][a-zа-яёA-ZА-ЯЁ0-9]{2,}(?:\s+[A-ZА-ЯЁ][a-zа-яёA-ZА-ЯЁ0-9]{2,}){0,2})\b"
)
_TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9]{3,}")


def normalize_entity(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def extract_entities(*texts: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for m in _ENTITY_RE.findall(text or ""):
            n = normalize_entity(m)
            if n and n not in seen and len(n) >= 3:
                seen.add(n)
                found.append(n)
        # also key-like tokens from fact keys: identity.name → skip; acme.project → acme
        for tok in _TOKEN_RE.findall(text or ""):
            if tok[0].isupper() or (len(tok) >= 5 and tok.isalpha()):
                n = normalize_entity(tok)
                if n not in seen and len(n) >= 4:
                    seen.add(n)
                    found.append(n)
    return found[:40]


@dataclass
class EntityNode:
    name: str
    fact_keys: set[str] = field(default_factory=set)
    related: set[str] = field(default_factory=set)


@dataclass
class EntityMemoryIndex:
    """Per-tenant entity graph."""
    nodes: dict[str, EntityNode] = field(default_factory=dict)

    def ingest_fact(self, key: str, value: str) -> list[str]:
        ents = extract_entities(key.replace(".", " "), value)
        # seed from key prefix segments
        for part in (key or "").split("."):
            if len(part) >= 4 and part.isalpha():
                ents.append(normalize_entity(part))
        ents = list(dict.fromkeys(ents))
        for e in ents:
            node = self.nodes.setdefault(e, EntityNode(name=e))
            node.fact_keys.add(key)
            for other in ents:
                if other != e:
                    node.related.add(other)
        return ents

    def expand_from_query(self, query: str, *, max_facts: int = 8) -> list[str]:
        """Return fact keys linked to entities mentioned in the query."""
        q_ents = extract_entities(query)
        if not q_ents:
            # fallback: any token match against node names
            toks = {t.lower() for t in _TOKEN_RE.findall(query or "")}
            q_ents = [n for n in self.nodes if any(t in n for t in toks)][:6]
        keys: list[str] = []
        seen: set[str] = set()
        for e in q_ents:
            node = self.nodes.get(normalize_entity(e)) or self.nodes.get(e)
            if not node:
                continue
            for k in node.fact_keys:
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
            for rel in list(node.related)[:4]:
                rn = self.nodes.get(rel)
                if not rn:
                    continue
                for k in rn.fact_keys:
                    if k not in seen:
                        seen.add(k)
                        keys.append(k)
            if len(keys) >= max_facts:
                break
        return keys[:max_facts]

    def to_payload(self) -> dict:
        return {
            name: {
                "facts": sorted(n.fact_keys)[:40],
                "related": sorted(n.related)[:20],
            }
            for name, n in list(self.nodes.items())[:200]
        }

    @classmethod
    def from_payload(cls, raw: dict) -> "EntityMemoryIndex":
        idx = cls()
        for name, blob in (raw or {}).items():
            node = EntityNode(
                name=name,
                fact_keys=set(blob.get("facts") or []),
                related=set(blob.get("related") or []),
            )
            idx.nodes[name] = node
        return idx


_INDEX_CACHE: dict[str, EntityMemoryIndex] = {}


def _cache_key(company_id: str, user_id: str) -> str:
    return f"{company_id}|{user_id}"


def get_entity_index(company_id: str, user_id: str) -> EntityMemoryIndex:
    return _INDEX_CACHE.setdefault(_cache_key(company_id, user_id), EntityMemoryIndex())


def rebuild_entity_index(company_id: str, user_id: str, facts: list[tuple[str, str]]) -> EntityMemoryIndex:
    idx = EntityMemoryIndex()
    for k, v in facts:
        idx.ingest_fact(k, v)
    _INDEX_CACHE[_cache_key(company_id, user_id)] = idx
    return idx


def persist_entity_index_redis(redis_client: Any, company_id: str, user_id: str, idx: EntityMemoryIndex, *, ttl: int = 86400) -> None:
    if redis_client is None:
        return
    try:
        key = f"finkey:entity_mem:{company_id}:{user_id}"
        redis_client.setex(key, ttl, json.dumps(idx.to_payload(), ensure_ascii=False))
    except Exception:
        pass


def load_entity_index_redis(redis_client: Any, company_id: str, user_id: str) -> Optional[EntityMemoryIndex]:
    if redis_client is None:
        return None
    try:
        key = f"finkey:entity_mem:{company_id}:{user_id}"
        raw = redis_client.get(key)
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return EntityMemoryIndex.from_payload(json.loads(raw))
    except Exception:
        return None
