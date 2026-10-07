# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""FinKeyMemory adapter: L0 substrate, extractor-free ingest of pre-extracted QA sessions."""
from __future__ import annotations

from finkey_memory import ExtractedFact
from finkey_memory.factory import build_memory_manager


class FinkeyAdapter:
    def __init__(self) -> None:
        self.mm = build_memory_manager()
        self._qa: dict[str, list[tuple[str, str]]] = {}

    def ingest(self, sessions: list[dict]) -> None:
        """
        sessions: [{user_id, turns: [{role, content}]}]
        Facts are asked of the dataset's own metadata when present
        (``turns[].fact`` = {key, value}); plain transcripts are kept as
        conversation lines for substring answering.
        """
        for s in sessions:
            uid = str(s.get("user_id") or "default")
            lines = self._qa.setdefault(uid, [])
            for t in s.get("turns", []):
                fact = t.get("fact")
                if isinstance(fact, dict) and fact.get("key") and fact.get("value"):
                    self.mm.upsert_extracted_facts(
                        company_id="eval", user_id=uid, conversation_id=None,
                        facts=[ExtractedFact(
                            key_normalized=str(fact["key"]),
                            value=str(fact["value"]),
                            source="user",
                        )],
                    )
                if t.get("role") and t.get("content"):
                    lines.append((str(t["role"]), str(t["content"])))

    def answer(self, question: str) -> str:
        best: list[str] = []
        for uid, lines in self._qa.items():
            ctx = self.mm.load_context("eval", uid, "qa", question)
            block = ctx.memory_prompt_block or ""
            if question.lower() in "\n".join(c for _, c in lines).lower() or block:
                best.append(block)
                best.extend(c for _, c in lines[-6:])
        return "\n".join(best)[:2000]
