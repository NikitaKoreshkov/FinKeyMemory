# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Лёгкие метрики и структурированные логи для слоя памяти (Phase 7).

Зачем не сразу ``prometheus_client``:
  * Пакет не тянет HTTP-зависимость; эндпоинт публикует приложение-обёртка.
  * В тестах удобно иметь in-process registry с детерминированным состоянием.
  * Когда ``prometheus_client`` всё-таки доступен — автоматически зеркалим в него,
    чтобы приложению достаточно было ``prometheus_client.exposition.generate_latest()``.

Что предоставляется:
  * ``Counter``, ``Gauge``, ``Histogram`` — потокобезопасные с лейблами.
  * Глобальный singleton ``METRICS`` — все метрики памяти регистрируются на нём.
  * ``expose_text()`` — Prometheus text format для собственного экспорта (без HTTP).
  * ``snapshot()`` — словарь {name: {labels-tuple: value}} для тестов и алёртов.
  * ``log_event(...)`` — структурированный лог с обязательными tenant-полями.

Дизайн-выбор по бакетам Histogram: 5 ms … 30 s, экспоненциально — хватает для
LLM-вызовов (extract), быстрых DB-операций (PG upsert) и медленных (Qdrant поиск).
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, Optional

logger = logging.getLogger("finkey.memory.metrics")



DEFAULT_BUCKETS: tuple[float, ...] = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
)



try:
    import prometheus_client as _prom  # type: ignore
    _PROM_AVAILABLE = True
except Exception:  # pragma: no cover
    _prom = None
    _PROM_AVAILABLE = False


def _prom_enabled() -> bool:
    """Можно жёстко отключить bridge через env (для тестов или dev-режима)."""
    if not _PROM_AVAILABLE:
        return False
    raw = (os.getenv("FINKEY_METRICS_PROMETHEUS_BRIDGE") or "").strip().lower()
    if raw in ("0", "false", "off", "no"):
        return False
    return True




def _normalize_labels(labels: Optional[Dict[str, Any]], allowed: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    """Сортируем лейблы по имени для стабильного ключа; неизвестные — отбрасываем."""
    if not labels:
        return tuple((k, "") for k in sorted(allowed))
    out = {k: ("" if labels.get(k) is None else str(labels.get(k))) for k in allowed}
    return tuple(sorted(out.items()))


@dataclass
class _CounterCell:
    value: float = 0.0


class Counter:
    """Монотонно растущий счётчик. Может иметь лейблы."""

    __slots__ = ("name", "help", "label_names", "_cells", "_lock", "_prom")

    def __init__(self, name: str, help: str, label_names: tuple[str, ...] = ()) -> None:  # noqa: A002 — name mirrors prometheus
        self.name = name
        self.help = help
        self.label_names = label_names
        self._cells: Dict[tuple[tuple[str, str], ...], _CounterCell] = {}
        self._lock = threading.Lock()
        self._prom = None
        if _prom_enabled():
            try:
                self._prom = _prom.Counter(name, help, label_names)
            except Exception as exc:  # pragma: no cover
                logger.debug("Counter prom bridge failed for %s: %s", name, exc)

    def inc(self, value: float = 1.0, **labels: Any) -> None:
        key = _normalize_labels(labels, self.label_names)
        with self._lock:
            cell = self._cells.get(key)
            if cell is None:
                cell = _CounterCell()
                self._cells[key] = cell
            cell.value += float(value)
        if self._prom is not None:
            try:
                if self.label_names:
                    self._prom.labels(**{k: labels.get(k, "") for k in self.label_names}).inc(value)
                else:
                    self._prom.inc(value)
            except Exception:  # pragma: no cover
                pass

    def collect(self) -> Iterable[tuple[tuple[tuple[str, str], ...], float]]:
        with self._lock:
            return [(k, c.value) for k, c in self._cells.items()]


@dataclass
class _GaugeCell:
    value: float = 0.0


class Gauge:
    """Значение, которое может расти и уменьшаться (in-flight, current size)."""

    __slots__ = ("name", "help", "label_names", "_cells", "_lock", "_prom")

    def __init__(self, name: str, help: str, label_names: tuple[str, ...] = ()) -> None:  # noqa: A002
        self.name = name
        self.help = help
        self.label_names = label_names
        self._cells: Dict[tuple[tuple[str, str], ...], _GaugeCell] = {}
        self._lock = threading.Lock()
        self._prom = None
        if _prom_enabled():
            try:
                self._prom = _prom.Gauge(name, help, label_names)
            except Exception as exc:  # pragma: no cover
                logger.debug("Gauge prom bridge failed for %s: %s", name, exc)

    def set(self, value: float, **labels: Any) -> None:
        key = _normalize_labels(labels, self.label_names)
        with self._lock:
            cell = self._cells.get(key)
            if cell is None:
                cell = _GaugeCell()
                self._cells[key] = cell
            cell.value = float(value)
        if self._prom is not None:
            try:
                if self.label_names:
                    self._prom.labels(**{k: labels.get(k, "") for k in self.label_names}).set(value)
                else:
                    self._prom.set(value)
            except Exception:  # pragma: no cover
                pass

    def inc(self, value: float = 1.0, **labels: Any) -> None:
        key = _normalize_labels(labels, self.label_names)
        with self._lock:
            cell = self._cells.get(key)
            if cell is None:
                cell = _GaugeCell()
                self._cells[key] = cell
            cell.value += float(value)
        if self._prom is not None:
            try:
                if self.label_names:
                    self._prom.labels(**{k: labels.get(k, "") for k in self.label_names}).inc(value)
                else:
                    self._prom.inc(value)
            except Exception:  # pragma: no cover
                pass

    def dec(self, value: float = 1.0, **labels: Any) -> None:
        self.inc(-value, **labels)

    def collect(self) -> Iterable[tuple[tuple[tuple[str, str], ...], float]]:
        with self._lock:
            return [(k, c.value) for k, c in self._cells.items()]


class _HistCell:
    __slots__ = ("buckets", "count", "sum", "bucket_counts")

    def __init__(self, buckets: tuple[float, ...]) -> None:
        self.buckets = buckets
        self.bucket_counts = [0] * (len(buckets) + 1)
        self.count = 0
        self.sum = 0.0


class Histogram:
    """Простой гистограмм-распределитель."""

    __slots__ = ("name", "help", "label_names", "buckets", "_cells", "_lock", "_prom")

    def __init__(
        self,
        name: str,
        help: str,                                              # noqa: A002
        label_names: tuple[str, ...] = (),
        buckets: tuple[float, ...] = DEFAULT_BUCKETS,
    ) -> None:
        self.name = name
        self.help = help
        self.label_names = label_names
        self.buckets = buckets
        self._cells: Dict[tuple[tuple[str, str], ...], _HistCell] = {}
        self._lock = threading.Lock()
        self._prom = None
        if _prom_enabled():
            try:
                self._prom = _prom.Histogram(name, help, label_names, buckets=buckets)
            except Exception as exc:  # pragma: no cover
                logger.debug("Histogram prom bridge failed for %s: %s", name, exc)

    def observe(self, value: float, **labels: Any) -> None:
        key = _normalize_labels(labels, self.label_names)
        v = float(value)
        with self._lock:
            cell = self._cells.get(key)
            if cell is None:
                cell = _HistCell(self.buckets)
                self._cells[key] = cell
            cell.count += 1
            cell.sum += v
            placed = False
            for i, ub in enumerate(self.buckets):
                if v <= ub:
                    cell.bucket_counts[i] += 1
                    placed = True
                    break
            if not placed:
                cell.bucket_counts[-1] += 1
        if self._prom is not None:
            try:
                if self.label_names:
                    self._prom.labels(**{k: labels.get(k, "") for k in self.label_names}).observe(v)
                else:
                    self._prom.observe(v)
            except Exception:  # pragma: no cover
                pass

    @contextmanager
    def time(self, **labels: Any) -> Iterator[None]:
        """Контекст-менеджер: ``with hist.time(): ...``."""
        t0 = time.time()
        try:
            yield
        finally:
            self.observe(time.time() - t0, **labels)

    def collect(self) -> Iterable[tuple[tuple[tuple[str, str], ...], _HistCell]]:
        with self._lock:
            return [(k, c) for k, c in self._cells.items()]




@dataclass
class _Registry:
    counters:   Dict[str, Counter]   = field(default_factory=dict)
    gauges:     Dict[str, Gauge]     = field(default_factory=dict)
    histograms: Dict[str, Histogram] = field(default_factory=dict)

    def counter(self, name: str, help: str, labels: tuple[str, ...] = ()) -> Counter:  # noqa: A002
        c = self.counters.get(name)
        if c is None:
            c = Counter(name, help, labels)
            self.counters[name] = c
        return c

    def gauge(self, name: str, help: str, labels: tuple[str, ...] = ()) -> Gauge:  # noqa: A002
        g = self.gauges.get(name)
        if g is None:
            g = Gauge(name, help, labels)
            self.gauges[name] = g
        return g

    def histogram(
        self,
        name: str,
        help: str,                                              # noqa: A002
        labels: tuple[str, ...] = (),
        buckets: tuple[float, ...] = DEFAULT_BUCKETS,
    ) -> Histogram:
        h = self.histograms.get(name)
        if h is None:
            h = Histogram(name, help, labels, buckets)
            self.histograms[name] = h
        return h

    def reset(self) -> None:
        """Только для тестов: чистим все cell-values, не сами объекты-метрик."""
        for c in self.counters.values():
            with c._lock:
                c._cells.clear()
        for g in self.gauges.values():
            with g._lock:
                g._cells.clear()
        for h in self.histograms.values():
            with h._lock:
                h._cells.clear()


REGISTRY = _Registry()



_T = ("company_id",)

EXTRACTOR_RUNS         = REGISTRY.counter("finkey_memory_extractor_runs_total",        "Total MemoryExtractor invocations.", _T)
EXTRACTOR_PARSE_FAIL   = REGISTRY.counter("finkey_memory_extractor_parse_failures_total","Extractor responses that failed JSON parse.", _T)
EXTRACTOR_LLM_ERRORS   = REGISTRY.counter("finkey_memory_extractor_llm_errors_total",   "Extractor LLM call failures.", _T)
EXTRACTOR_FACTS        = REGISTRY.counter("finkey_memory_extractor_facts_total",        "Facts returned by extractor (sum).", _T)
EXTRACTOR_LATENCY      = REGISTRY.histogram("finkey_memory_extractor_latency_seconds",   "Wall time per extractor call (LLM + parse).", _T)

RUNNER_SCHEDULED       = REGISTRY.counter("finkey_memory_runner_scheduled_total",  "AsyncExtractorRunner tasks scheduled.", _T)
RUNNER_SKIPPED_LOCKED  = REGISTRY.counter("finkey_memory_runner_skipped_locked_total","Runner skipped due to active lock.", _T)
RUNNER_SKIPPED_DEBOUNCE= REGISTRY.counter("finkey_memory_runner_skipped_debounce_total","Runner skipped due to debounce schedule.", _T)
RUNNER_COMPLETED       = REGISTRY.counter("finkey_memory_runner_completed_total",  "Runner tasks completed (parse ok or empty).", _T)
RUNNER_TIMEOUTS        = REGISTRY.counter("finkey_memory_runner_timeouts_total",   "Runner tasks that breached timeout.", _T)
RUNNER_ERRORS          = REGISTRY.counter("finkey_memory_runner_errors_total",     "Runner tasks that crashed.", _T)
RUNNER_INFLIGHT        = REGISTRY.gauge  ("finkey_memory_runner_inflight",         "Currently running extractor tasks.", _T)

FACT_UPSERT_TOTAL      = REGISTRY.counter("finkey_memory_fact_upsert_total",       "Facts pushed to FactStore (any backend).", _T)
FACT_UPSERT_PG_OK      = REGISTRY.counter("finkey_memory_fact_upsert_pg_ok_total", "Facts successfully written to PostgreSQL.", _T)
FACT_UPSERT_PG_FAIL    = REGISTRY.counter("finkey_memory_fact_upsert_pg_fail_total","Facts that failed PostgreSQL upsert.", _T)
FACT_UPSERT_QDRANT_OK  = REGISTRY.counter("finkey_memory_fact_upsert_qdrant_ok_total","Facts successfully indexed into Qdrant.", _T)
FACT_UPSERT_QDRANT_FAIL= REGISTRY.counter("finkey_memory_fact_upsert_qdrant_fail_total","Facts that failed Qdrant indexing.", _T)
FACT_DELETES           = REGISTRY.counter("finkey_memory_fact_deletes_total",      "soft_delete_fact invocations.", _T)
FACT_PURGES            = REGISTRY.counter("finkey_memory_fact_purge_total",        "purge-user (memory.purge_user) invocations.", _T)

LOAD_CONTEXT_LATENCY   = REGISTRY.histogram("finkey_memory_load_context_seconds", "MemoryManager.load_context wall time.", _T)




def snapshot() -> dict:
    """
    Возвращает чистый дикт состояния — удобно в тестах и в вашем HTTP-эндпоинте.

    Формат:
      {
        "counters":   { name: { labels_tuple: value } },
        "gauges":     { name: { labels_tuple: value } },
        "histograms": { name: { labels_tuple: {"count": int, "sum": float, "buckets": [...]} } },
      }
    """
    out: dict[str, dict] = {"counters": {}, "gauges": {}, "histograms": {}}
    for c in REGISTRY.counters.values():
        out["counters"][c.name] = {k: v for k, v in c.collect()}
    for g in REGISTRY.gauges.values():
        out["gauges"][g.name] = {k: v for k, v in g.collect()}
    for h in REGISTRY.histograms.values():
        cells: dict = {}
        for k, cell in h.collect():
            cells[k] = {
                "count":   cell.count,
                "sum":     cell.sum,
                "buckets": list(zip(h.buckets, cell.bucket_counts[:-1])) + [(math.inf, cell.bucket_counts[-1])],
            }
        out["histograms"][h.name] = cells
    return out


def _fmt_labels(labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return ""
    parts = [f'{k}="{_escape(v)}"' for k, v in labels]
    return "{" + ",".join(parts) + "}"


def _escape(v: str) -> str:
    return v.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def expose_text() -> str:
    """
    Сборка Prometheus-совместимого текста (text/plain v0.0.4).

    Если установлен ``prometheus_client`` и bridge включён — рекомендую
    использовать ``prometheus_client.exposition.generate_latest()`` напрямую,
    он включает gauge/histogram/summary с server-side нагрузкой.
    Эта же функция — чистый Python fallback (например, для unit-тестов).
    """
    lines: list[str] = []

    for c in REGISTRY.counters.values():
        lines.append(f"# HELP {c.name} {c.help}")
        lines.append(f"# TYPE {c.name} counter")
        for labels, val in c.collect():
            lines.append(f"{c.name}{_fmt_labels(labels)} {val}")

    for g in REGISTRY.gauges.values():
        lines.append(f"# HELP {g.name} {g.help}")
        lines.append(f"# TYPE {g.name} gauge")
        for labels, val in g.collect():
            lines.append(f"{g.name}{_fmt_labels(labels)} {val}")

    for h in REGISTRY.histograms.values():
        lines.append(f"# HELP {h.name} {h.help}")
        lines.append(f"# TYPE {h.name} histogram")
        for labels, cell in h.collect():
            cum = 0
            for ub, bc in zip(h.buckets, cell.bucket_counts[:-1]):
                cum += bc
                extra = ((("le", str(ub)),))
                lines.append(f"{h.name}_bucket{_fmt_labels(tuple(list(labels) + list(extra)))} {cum}")
            cum += cell.bucket_counts[-1]
            inf_extra = ((("le", "+Inf"),))
            lines.append(f"{h.name}_bucket{_fmt_labels(tuple(list(labels) + list(inf_extra)))} {cum}")
            lines.append(f"{h.name}_sum{_fmt_labels(labels)} {cell.sum}")
            lines.append(f"{h.name}_count{_fmt_labels(labels)} {cell.count}")

    return "\n".join(lines) + ("\n" if lines else "")




def log_event(
    logger_obj: logging.Logger,
    level: int,
    event: str,
    *,
    company_id: Optional[str] = None,
    user_id:    Optional[str] = None,
    conv_id:    Optional[str] = None,
    **fields: Any,
) -> None:
    """
    Структурированный лог: ``logger.log(level, event, extra={...})``.

    Жёстко гарантирует наличие tenant-полей (``company_id``, ``user_id``, ``conv_id``)
    в ``extra``, остальное — произвольные поля. Сторонние handlers (json-formatter,
    OTel-exporter) увидят их единообразно.
    """
    extra: dict[str, Any] = {
        "fk_event":      event,
        "fk_company_id": company_id or "",
        "fk_user_id":    user_id or "",
        "fk_conv_id":    conv_id or "",
    }
    for k, v in fields.items():
        extra[f"fk_{k}"] = v
    try:
        logger_obj.log(level, event, extra=extra)
    except Exception:  # pragma: no cover
        pass
