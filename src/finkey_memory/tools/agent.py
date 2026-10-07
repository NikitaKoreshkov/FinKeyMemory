# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Optional preflight: one short chat.completions round with ``MEMORY_TOOLS``.

The **main** model decides via ``tool_choice: auto`` whether to call tools.
Simple turns → no ``tool_calls`` → zero extra DB work beyond the LLM stub call.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

from finkey_memory.tools.dispatcher import MemoryToolDispatcher
from finkey_memory.tools.schema import MEMORY_TOOLS

logger = logging.getLogger(__name__)

CompleterFn = Callable[..., dict]


@dataclass
class MemoryToolBundle:
    """Structured outcome for host-app prompt injection."""

    block_text: str
    had_tool_calls: bool
    raw_tool_messages: list[dict[str, Any]]


_MEMORY_ROUTER_SYSTEM = """You are FinKey's memory-routing assistant (internal).
You MAY call the provided memory_* tools when the user's latest message clearly benefits from:
- retrieving stored personal facts, past chat themes, or company knowledge base chunks, OR
- saving / deleting / listing durable user facts.

If the message is small talk, gratitude, generic questions answerable without stored memory,
or the user did not imply any need for recall or persistence — respond with exactly:
PASS
and do NOT call any tools.

Rules:
- Never invent stored facts; use tools to read/write truth.
- Keep any free-text reply besides PASS minimal (one line)."""


class MemoryToolAgent:
    """Runs a bounded tool loop (non-streaming) against an OpenAI-compatible API."""

    def __init__(
        self,
        *,
        max_rounds: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> None:
        self._max_rounds = max_rounds if max_rounds is not None else max(
            1,
            int(os.getenv("FINKEY_MEMORY_TOOLS_MAX_ROUNDS", "4") or "4"),
        )
        self._max_tokens = max_tokens if max_tokens is not None else max(
            64,
            int(os.getenv("FINKEY_MEMORY_TOOLS_MAX_TOKENS", "512") or "512"),
        )
        self._temperature = temperature

    def run(
        self,
        *,
        user_message: str,
        dispatcher: MemoryToolDispatcher,
        completer: CompleterFn,
    ) -> MemoryToolBundle:
        """
        ``completer`` must behave like ``OpenRouterProvider.chat_completion_json``:
        ``(messages=..., tools=..., temperature=..., max_tokens=...) -> dict`` (full JSON body).
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _MEMORY_ROUTER_SYSTEM},
            {
                "role": "user",
                "content": (
                    "Latest user message:\n"
                    f"{user_message.strip()}\n\n"
                    "Decide whether to call memory tools or respond PASS."
                ),
            },
        ]
        had_tool_calls = False
        raw_trail: list[dict[str, Any]] = []

        for _ in range(self._max_rounds):
            try:
                data = completer(
                    messages=messages,
                    tools=MEMORY_TOOLS,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
            except Exception as exc:
                logger.warning("Memory tool preflight LLM failed: %s", exc)
                return MemoryToolBundle(block_text="", had_tool_calls=False, raw_tool_messages=[])

            choices = data.get("choices") or []
            msg = (choices[0].get("message") if choices else None) or {}
            tool_calls = msg.get("tool_calls")

            if tool_calls:
                had_tool_calls = True
                am: dict[str, Any] = {"role": "assistant"}
                if msg.get("content") is not None:
                    am["content"] = msg.get("content")
                if msg.get("tool_calls"):
                    am["tool_calls"] = msg["tool_calls"]
                messages.append(am)
                raw_trail.append({"assistant": am})

                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    tid = tc.get("id") or ""
                    fn_obj = tc.get("function") or {}
                    fn = fn_obj.get("name") or ""
                    args_raw = fn_obj.get("arguments") or "{}"
                    try:
                        args = json.loads(args_raw) if isinstance(args_raw, str) else {}
                    except json.JSONDecodeError:
                        args = {}
                    if not isinstance(args, dict):
                        args = {}
                    result = dispatcher.execute(fn, args)
                    raw_trail.append({"tool": fn, "tool_call_id": tid, "result_preview": result[:800]})
                    messages.append({"role": "tool", "tool_call_id": tid, "content": result})
                continue

            break

        if not had_tool_calls:
            return MemoryToolBundle(block_text="", had_tool_calls=False, raw_tool_messages=[])

        tool_payloads: list[Any] = []
        for m in messages:
            if m.get("role") == "tool":
                try:
                    tool_payloads.append(json.loads(m.get("content") or "{}"))
                except json.JSONDecodeError:
                    tool_payloads.append({"raw": (m.get("content") or "")[:2000]})

        block = (
            "### Memory tools (model-initiated)\n"
            "Structured tool results from a pre-turn memory pass. "
            "Use when relevant; do not contradict stored facts without asking the user.\n"
            "```json\n"
            f"{json.dumps(tool_payloads, ensure_ascii=False, indent=2)[:12000]}\n"
            "```\n"
        )
        return MemoryToolBundle(
            block_text=block,
            had_tool_calls=True,
            raw_tool_messages=raw_trail,
        )
