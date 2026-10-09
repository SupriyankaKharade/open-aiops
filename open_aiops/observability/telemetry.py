from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter
from opentelemetry.trace import Span, StatusCode, Tracer

_TRACER_NAME = "open_aiops.telemetry"


def get_tracer() -> Tracer:
    """Get the active OpenTelemetry tracer."""
    return trace.get_tracer(_TRACER_NAME)


def init_telemetry(
    service_name: str = "open-aiops",
    exporter: Optional[SpanExporter] = None,
    force_reset: bool = False,
) -> Tracer:
    """Initialize OpenTelemetry TracerProvider and set global tracer."""
    if force_reset:
        trace._TRACER_PROVIDER = None
        if hasattr(trace, "_TRACER_PROVIDER_SET_ONCE"):
            trace._TRACER_PROVIDER_SET_ONCE._done = False

    if force_reset or not isinstance(trace.get_tracer_provider(), TracerProvider):
        provider = TracerProvider()
        if exporter:
            provider.add_span_processor(SimpleSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

    return get_tracer()


def start_span(
    name: str,
    model_name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> Span:
    """Start an OpenTelemetry span with GenAI attributes."""
    tracer = get_tracer()
    merged_attrs = attributes.copy() if attributes else {}

    if model_name:
        merged_attrs["gen_ai.request.model"] = model_name

    return tracer.start_span(name=name, attributes=merged_attrs)


def end_span(
    span: Span,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    error: Optional[Exception] = None,
    additional_attributes: Optional[Dict[str, Any]] = None,
) -> None:
    """End a span and record token metrics and execution status."""
    if additional_attributes:
        for key, val in additional_attributes.items():
            span.set_attribute(key, val)

    if input_tokens is not None:
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)

    if output_tokens is not None:
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)

    if total_tokens is not None:
        span.set_attribute("gen_ai.usage.total_tokens", total_tokens)
    elif input_tokens is not None and output_tokens is not None:
        span.set_attribute("gen_ai.usage.total_tokens", input_tokens + output_tokens)

    if error:
        span.record_exception(error)
        span.set_status(StatusCode.ERROR, description=str(error))
    else:
        span.set_status(StatusCode.OK)

    span.end()


@contextmanager
def trace_llm_call(
    name: str = "llm.request",
    model_name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> Generator[Dict[str, Any], None, None]:
    """Context manager for tracing model calls with token metrics capture."""
    span = start_span(name=name, model_name=model_name, attributes=attributes)
    context_data: Dict[str, Any] = {}
    try:
        yield context_data
    except Exception as exc:
        end_span(
            span,
            input_tokens=context_data.get("input_tokens"),
            output_tokens=context_data.get("output_tokens"),
            total_tokens=context_data.get("total_tokens"),
            error=exc,
        )
        raise
    else:
        end_span(
            span,
            input_tokens=context_data.get("input_tokens"),
            output_tokens=context_data.get("output_tokens"),
            total_tokens=context_data.get("total_tokens"),
        )
