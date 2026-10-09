import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from open_aiops.observability.telemetry import (
    end_span,
    init_telemetry,
    start_span,
    trace_llm_call,
)


@pytest.fixture
def exporter():
    exp = InMemorySpanExporter()
    init_telemetry(service_name="test-runner", exporter=exp, force_reset=True)
    return exp


def test_start_and_end_span(exporter):
    span = start_span("test-operation", model_name="gpt-4o", attributes={"custom.key": "val"})
    end_span(span, input_tokens=10, output_tokens=20)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    finished = spans[0]
    assert finished.name == "test-operation"
    assert finished.attributes["gen_ai.request.model"] == "gpt-4o"
    assert finished.attributes["custom.key"] == "val"
    assert finished.attributes["gen_ai.usage.input_tokens"] == 10
    assert finished.attributes["gen_ai.usage.output_tokens"] == 20
    assert finished.attributes["gen_ai.usage.total_tokens"] == 30
    assert finished.status.status_code == StatusCode.OK


def test_end_span_with_error(exporter):
    span = start_span("failing-op")
    err = ValueError("API timeout")
    end_span(span, error=err)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR
    assert spans[0].status.description == "API timeout"


def test_trace_llm_call_context_manager(exporter):
    with trace_llm_call("call-gpt", model_name="claude-3-5-sonnet") as ctx:
        ctx["input_tokens"] = 100
        ctx["output_tokens"] = 50

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "call-gpt"
    assert spans[0].attributes["gen_ai.request.model"] == "claude-3-5-sonnet"
    assert spans[0].attributes["gen_ai.usage.total_tokens"] == 150


def test_trace_llm_call_with_exception(exporter):
    with pytest.raises(RuntimeError, match="Network failure"):
        with trace_llm_call("failing-llm", model_name="gpt-4o") as ctx:
            ctx["input_tokens"] = 25
            raise RuntimeError("Network failure")

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "failing-llm"
    assert spans[0].status.status_code == StatusCode.ERROR
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 25


def test_end_span_explicit_total_and_additional_attributes(exporter):
    span = start_span("custom-span")
    end_span(
        span,
        total_tokens=99,
        additional_attributes={"custom.tag": "production", "user.id": "user-123"},
    )

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["gen_ai.usage.total_tokens"] == 99
    assert spans[0].attributes["custom.tag"] == "production"
    assert spans[0].attributes["user.id"] == "user-123"
