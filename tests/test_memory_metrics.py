# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Unit-тесты Phase 7: Prometheus-метрики и структурированные логи слоя памяти.

Покрываются:
  * базовые операции Counter/Gauge/Histogram,
  * lazy-инициализация cell-ей для лейблов,
  * snapshot() возвращает консистентный дикт,
  * expose_text() — Prometheus text-format с правильными HELP/TYPE/buckets/_sum/_count,
  * log_event() гарантированно добавляет tenant-поля в record.extra,
  * интеграция: MemoryExtractor.extract() инкрементирует runs/parse_failures/llm_errors.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

import pytest

from finkey_memory import metrics as M


@pytest.fixture(autouse=True)
def _reset_registry():
    M.REGISTRY.reset()
    yield
    M.REGISTRY.reset()




def test_counter_increments_per_label_independently() -> None:
    c = M.REGISTRY.counter("finkey_test_counter", "help", ("company_id",))
    c.inc(company_id="acme")
    c.inc(2.5, company_id="acme")
    c.inc(company_id="globex")

    cells = dict(c.collect())
    acme_key = (("company_id", "acme"),)
    globex_key = (("company_id", "globex"),)
    assert cells[acme_key] == pytest.approx(3.5)
    assert cells[globex_key] == pytest.approx(1.0)


def test_counter_without_labels_uses_empty_key() -> None:
    c = M.REGISTRY.counter("finkey_test_counter_nolab", "help")
    c.inc()
    c.inc(4)
    cells = dict(c.collect())
    assert cells[()] == pytest.approx(5.0)


def test_gauge_inc_dec_and_set() -> None:
    g = M.REGISTRY.gauge("finkey_test_gauge", "help", ("company_id",))
    g.inc(company_id="acme")
    g.inc(company_id="acme")
    g.dec(company_id="acme")
    g.set(7, company_id="globex")

    cells = dict(g.collect())
    assert cells[(("company_id", "acme"),)] == pytest.approx(1.0)
    assert cells[(("company_id", "globex"),)] == pytest.approx(7.0)


def test_histogram_observes_and_bucketizes() -> None:
    h = M.REGISTRY.histogram(
        "finkey_test_hist", "help", ("company_id",), buckets=(0.1, 1.0, 5.0),
    )
    for v in (0.05, 0.5, 2.0, 10.0):
        h.observe(v, company_id="acme")

    cells = dict(h.collect())
    cell = cells[(("company_id", "acme"),)]
    assert cell.count == 4
    assert cell.sum == pytest.approx(12.55)
    assert cell.bucket_counts == [1, 1, 1, 1]


def test_histogram_time_context_records_positive_duration() -> None:
    import time
    h = M.REGISTRY.histogram("finkey_test_hist_time", "help", ("company_id",))
    with h.time(company_id="acme"):
        time.sleep(0.001)
    cells = dict(h.collect())
    cell = cells[(("company_id", "acme"),)]
    assert cell.count == 1
    assert cell.sum > 0.0




def test_snapshot_has_expected_top_level_shape() -> None:
    snap = M.snapshot()
    assert set(snap.keys()) == {"counters", "gauges", "histograms"}
    assert "finkey_memory_extractor_runs_total" in snap["counters"]
    assert "finkey_memory_runner_inflight" in snap["gauges"]
    assert "finkey_memory_extractor_latency_seconds" in snap["histograms"]


def test_expose_text_renders_prometheus_format() -> None:
    M.EXTRACTOR_RUNS.inc(company_id="acme")
    M.EXTRACTOR_LATENCY.observe(0.42, company_id="acme")

    text = M.expose_text()

    assert "# HELP finkey_memory_extractor_runs_total" in text
    assert "# TYPE finkey_memory_extractor_runs_total counter" in text
    assert 'finkey_memory_extractor_runs_total{company_id="acme"} 1.0' in text

    assert "# TYPE finkey_memory_extractor_latency_seconds histogram" in text
    assert 'finkey_memory_extractor_latency_seconds_sum{company_id="acme"} 0.42' in text
    assert 'finkey_memory_extractor_latency_seconds_count{company_id="acme"} 1' in text
    assert 'le="+Inf"' in text


def test_expose_text_escapes_label_values() -> None:
    c = M.REGISTRY.counter("finkey_test_escape", "h", ("company_id",))
    c.inc(company_id='ev"il\nnewline\\')
    text = M.expose_text()
    assert 'company_id="ev\\"il\\nnewline\\\\"' in text




class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_log_event_attaches_tenant_extras() -> None:
    log = logging.getLogger("finkey.test.metrics")
    h = _CaptureHandler()
    h.setLevel(logging.DEBUG)
    log.setLevel(logging.DEBUG)
    log.addHandler(h)
    try:
        M.log_event(
            log, logging.INFO, "memory.test.event",
            company_id="acme", user_id="u-1", conv_id="c-1",
            facts=3, parse_ok=True,
        )
    finally:
        log.removeHandler(h)

    assert len(h.records) == 1
    rec = h.records[0]
    assert rec.getMessage() == "memory.test.event"
    assert getattr(rec, "fk_event") == "memory.test.event"
    assert getattr(rec, "fk_company_id") == "acme"
    assert getattr(rec, "fk_user_id") == "u-1"
    assert getattr(rec, "fk_conv_id") == "c-1"
    assert getattr(rec, "fk_facts") == 3
    assert getattr(rec, "fk_parse_ok") is True


def test_log_event_fills_missing_tenant_fields_with_empty_strings() -> None:
    log = logging.getLogger("finkey.test.metrics")
    h = _CaptureHandler()
    log.addHandler(h)
    try:
        M.log_event(log, logging.INFO, "memory.test.no_tenant")
    finally:
        log.removeHandler(h)

    rec = h.records[-1]
    assert getattr(rec, "fk_company_id") == ""
    assert getattr(rec, "fk_user_id") == ""
    assert getattr(rec, "fk_conv_id") == ""




def _make_extractor(completer) -> Any:
    from finkey_memory.extractor import ExtractorConfig, MemoryExtractor
    cfg = ExtractorConfig(
        enabled       = True,
        every_n_turns = 1,
        model_label   = "fake-llm",
        max_tokens    = 128,
        temperature   = 0.0,
        min_confidence= 0.0,
        max_facts     = 5,
    )
    return MemoryExtractor(completer=completer, config=cfg)


def test_extractor_increments_runs_and_facts_on_success() -> None:
    payload = (
        '{"facts": [{"category":"identity","key":"name","value":"Никита",'
        '"confidence":0.95,"evidence":"меня зовут Никита"}]}'
    )
    ext = _make_extractor(lambda *a, **kw: payload)
    res = ext.extract(
        company_id="acme", user_id="u-1", conversation_id="c-1",
        recent_turns=[{"role": "user", "content": "меня зовут Никита"}],
        known_keys=set(),
    )
    assert res.parse_ok is True
    assert len(res.facts) == 1

    snap = M.snapshot()
    assert snap["counters"]["finkey_memory_extractor_runs_total"][(("company_id", "acme"),)] == 1.0
    assert snap["counters"]["finkey_memory_extractor_facts_total"][(("company_id", "acme"),)] == 1.0
    hist = snap["histograms"]["finkey_memory_extractor_latency_seconds"]
    assert hist[(("company_id", "acme"),)]["count"] == 1


def test_extractor_increments_parse_failures_on_bad_json() -> None:
    ext = _make_extractor(lambda *a, **kw: "not a json at all")
    res = ext.extract(
        company_id="acme", user_id="u-1", conversation_id="c-1",
        recent_turns=[{"role": "user", "content": "hi"}],
        known_keys=set(),
    )
    assert res.parse_ok is False
    snap = M.snapshot()
    assert snap["counters"]["finkey_memory_extractor_runs_total"][(("company_id", "acme"),)] == 1.0
    assert snap["counters"]["finkey_memory_extractor_parse_failures_total"][(("company_id", "acme"),)] == 1.0


def test_extractor_increments_llm_errors_on_completer_exception() -> None:
    def boom(*_a, **_kw):
        raise RuntimeError("boom")

    ext = _make_extractor(boom)
    res = ext.extract(
        company_id="acme", user_id="u-1", conversation_id="c-1",
        recent_turns=[{"role": "user", "content": "hi"}],
        known_keys=set(),
    )
    assert res.parse_ok is False
    snap = M.snapshot()
    assert snap["counters"]["finkey_memory_extractor_llm_errors_total"][(("company_id", "acme"),)] == 1.0
    runs = snap["counters"]["finkey_memory_extractor_runs_total"].get((("company_id", "acme"),), 0.0)
    assert runs == 0.0




def test_runner_increments_debounce_when_skipping_turn() -> None:
    from finkey_memory.extractor import ExtractorConfig, MemoryExtractor
    from finkey_memory.extractor_runner import AsyncExtractorRunner

    ext = MemoryExtractor(
        completer=lambda *a, **kw: '{"facts": []}',
        config=ExtractorConfig(enabled=True, every_n_turns=3, model_label="fake"),
    )
    runner = AsyncExtractorRunner(
        extractor             = ext,
        recent_turns_provider = lambda *_a, **_kw: [],
        known_keys_provider   = lambda *_a, **_kw: set(),
        on_result             = lambda *_a, **_kw: None,
        redis_client          = None,
        max_workers           = 1,
        timeout_sec           = 1.0,
    )

    accepted = runner.schedule(
        company_id="acme", user_id="u-1", conversation_id="c-1",
        turn_index=2, fallback_recent_turns=[{"role": "user", "content": "x"}],
    )
    runner.shutdown(wait=False)

    assert accepted is False
    snap = M.snapshot()
    counter = snap["counters"]["finkey_memory_runner_skipped_debounce_total"]
    assert counter[(("company_id", "acme"),)] == 1.0
