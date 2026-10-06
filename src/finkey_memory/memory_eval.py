# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Offline memory quality harness (LoCoMo / LongMemEval inspired, lightweight).

Measures:
  * fact_hit — retrieved facts contain expected key/value needles
  * refuse_ok — unrelated queries do not leak forbidden needles
  * temporal_ok — as_of filtering respects validity windows
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Mapping, Optional, Sequence

from finkey_memory.temporal import fact_valid_at


@dataclass(frozen=True)
class MemoryEvalCase:
    query: str
    expect_any: tuple[str, ...] = ()
    forbid_any: tuple[str, ...] = ()
    must_refuse: bool = False
    as_of: Optional[datetime] = None
    kind: str = "recall"


@dataclass
class MemoryEvalResult:
    n: int = 0
    hit: float = 0.0
    refuse_ok: float = 0.0
    temporal_ok: float = 0.0
    missed: list[dict] = field(default_factory=list)


def run_memory_eval(
    cases: Sequence[MemoryEvalCase],
    recall_fn: Callable[[str], Sequence[str]],
    *,
    fact_rows_fn: Optional[Callable[[], Sequence[Mapping]]] = None,
) -> MemoryEvalResult:
    """
    ``recall_fn(query) -> list[str]`` returns texts injected into the prompt.
    Optional ``fact_rows_fn`` used for temporal_ok cases.
    """
    hits = []
    refuses = []
    temporals = []
    missed = []
    for case in cases:
        texts = list(recall_fn(case.query) or [])
        blob = "\n".join(texts).lower()
        if case.kind == "temporal" and fact_rows_fn and case.as_of is not None:
            rows = [r for r in fact_rows_fn() if fact_valid_at(dict(r), case.as_of)]
            blob = "\n".join(str(r.get("value") or "") for r in rows).lower()
            ok = True
            if case.expect_any:
                ok = any(x.lower() in blob for x in case.expect_any)
            if case.forbid_any and any(x.lower() in blob for x in case.forbid_any):
                ok = False
            temporals.append(1 if ok else 0)
            if not ok:
                missed.append({"query": case.query, "kind": "temporal"})
            continue

        if case.must_refuse or case.kind == "unrelated":
            leaked = [x for x in case.forbid_any if x.lower() in blob]
            ok = not leaked
            refuses.append(1 if ok else 0)
            if not ok:
                missed.append({"query": case.query, "kind": "refuse", "leaked": leaked})
            continue

        ok = True
        if case.expect_any:
            ok = any(x.lower() in blob for x in case.expect_any)
        if case.forbid_any and any(x.lower() in blob for x in case.forbid_any):
            ok = False
        hits.append(1 if ok else 0)
        if not ok:
            missed.append({"query": case.query, "kind": "recall"})

    def _avg(xs: list[int]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return MemoryEvalResult(
        n=len(cases),
        hit=_avg(hits),
        refuse_ok=_avg(refuses),
        temporal_ok=_avg(temporals),
        missed=missed,
    )
