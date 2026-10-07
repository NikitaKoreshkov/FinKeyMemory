# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
FinKey memory tools — OpenAI-format function calling for model-driven memory.

* ``schema.MEMORY_TOOLS`` — tool definitions.
* ``dispatcher.MemoryToolDispatcher`` — executes tool calls (PG + Qdrant + RAG).
* ``agent.MemoryToolAgent`` — short preflight with ``tool_choice=auto`` (host app wires LLM).

See module docstring in ``schema.py`` / ``agent.py`` for design notes.
"""

from finkey_memory.tools.agent import MemoryToolAgent, MemoryToolBundle
from finkey_memory.tools.dispatcher import MemoryToolDispatcher, MemoryToolError
from finkey_memory.tools.schema import MEMORY_TOOL_NAMES, MEMORY_TOOLS

__all__ = [
    "MEMORY_TOOL_NAMES",
    "MEMORY_TOOLS",
    "MemoryToolAgent",
    "MemoryToolBundle",
    "MemoryToolDispatcher",
    "MemoryToolError",
]
