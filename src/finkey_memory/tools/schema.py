# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""OpenAI-format tool definitions for model-driven memory access.

Preflight runs in the llm-gateway in parallel with web search and MindEngine; the
chat model decides via ``tool_choice: auto`` whether to invoke any tool — no
language-specific trigger lists.
"""

from __future__ import annotations

MEMORY_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": (
                "Semantic search over durable user facts, company knowledge base (RAG), "
                "past conversation summaries, and FinKey user impressions. "
                "Use when the user asks about something that may be stored across sessions "
                "or in uploaded company documents. Omit scopes to search all enabled layers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query in the user's language (or English).",
                    },
                    "scopes": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "facts",
                                "company_kb",
                                "conversation_history",
                                "user_impressions",
                            ],
                        },
                        "description": "Which memory layers to query. Default: all.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Max hits per layer (default 8, max 24).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_save_fact",
            "description": (
                "Persist a durable fact about the current user (identity, role, project, "
                "preference, goal, etc.). Use when the user explicitly asks to remember something "
                "or states stable personal/work context worth storing. key_normalized must be "
                "ASCII snake_case, e.g. identity.name, context.current_project."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key_normalized": {"type": "string"},
                    "value": {"type": "string"},
                    "category": {
                        "type": "string",
                        "description": (
                            "One of: identity, affiliation, context, preference, goal, "
                            "constraint, commitment, relationship, state, other"
                        ),
                    },
                    "confidence": {
                        "type": "number",
                        "description": "0–1, default 0.85 for explicit user statements.",
                    },
                    "evidence_snippet": {
                        "type": "string",
                        "description": "Short verbatim excerpt from the user message (optional).",
                    },
                },
                "required": ["key_normalized", "value", "category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_forget_fact",
            "description": (
                "Soft-delete one durable fact by key_normalized (right to be forgotten / "
                "correction). Use when the user asks to forget a stored item."
            ),
            "parameters": {
                "type": "object",
                "properties": {"key_normalized": {"type": "string"}},
                "required": ["key_normalized"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_list_facts",
            "description": (
                "List active durable facts for the current user from the database "
                "(structured). Use for 'what do you know about me' or to pick keys before forget."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional filter by category.",
                    },
                    "limit": {"type": "integer", "description": "Max rows (default 40, cap 80)."},
                },
                "required": [],
            },
        },
    },
]

MEMORY_TOOL_NAMES: frozenset[str] = frozenset(
    t["function"]["name"] for t in MEMORY_TOOLS
)
