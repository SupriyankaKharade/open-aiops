"""End-to-end multi-agent demonstration script for OpenAIOps.

Simulates multi-agent requests utilizing OpenTelemetry tracing, dynamic model routing,
and automated failover handling.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure open_aiops package is resolvable when script is executed directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from open_aiops.governance.router import (
    ModelRouter,
    NoProviderAvailable,
    ProviderConfig,
    RouteRequest,
    estimate_cost,
)
from open_aiops.observability.telemetry import (
    init_telemetry,
    trace_llm_call,
)


def create_demo_router() -> ModelRouter:
    """Create a ModelRouter with representative LLM providers."""
    providers = [
        ProviderConfig(
            name="cloud-primary",
            model="gpt-4o",
            cost_per_1k_input_tokens=0.002,
            cost_per_1k_output_tokens=0.006,
            expected_latency_ms=350.0,
        ),
        ProviderConfig(
            name="cloud-fallback",
            model="claude-3-5-sonnet",
            cost_per_1k_input_tokens=0.004,
            cost_per_1k_output_tokens=0.010,
            expected_latency_ms=500.0,
        ),
        ProviderConfig(
            name="local-economy",
            model="llama-3-70b-instruct",
            cost_per_1k_input_tokens=0.0005,
            cost_per_1k_output_tokens=0.0008,
            expected_latency_ms=900.0,
        ),
    ]
    return ModelRouter(providers)


def simulate_agent_step(
    router: ModelRouter,
    agent_name: str,
    request: RouteRequest,
    simulated_output_tokens: int,
    fail_provider: bool = False,
) -> Dict[str, Any]:
    """Execute a single agent request with telemetry tracing and routing."""
    # 1. Attempt to route the request
    try:
        selected_provider = router.route(request)
    except NoProviderAvailable as exc:
        return {
            "agent": agent_name,
            "success": False,
            "error": str(exc),
            "provider": None,
        }

    est_cost = estimate_cost(selected_provider, request)

    # 2. Trace the execution lifecycle with OpenTelemetry
    span_name = f"agent.{agent_name}.call"
    execution_status = "OK"

    try:
        with trace_llm_call(
            name=span_name,
            model_name=selected_provider.model,
            attributes={
                "agent.name": agent_name,
                "provider.name": selected_provider.name,
                "cost.estimated": est_cost,
            },
        ) as span_ctx:
            if fail_provider:
                # Simulate a provider failure (e.g. HTTP 429 Rate Limit)
                raise RuntimeError(
                    f"Rate limit exceeded (HTTP 429) on {selected_provider.name}"
                )

            # Record simulated token usage
            span_ctx["input_tokens"] = request.input_tokens
            span_ctx["output_tokens"] = simulated_output_tokens
            span_ctx["total_tokens"] = request.input_tokens + simulated_output_tokens
    except RuntimeError as err:
        execution_status = f"FAILED: {err}"
        return {
            "agent": agent_name,
            "success": False,
            "error": str(err),
            "provider": selected_provider.name,
            "model": selected_provider.model,
            "estimated_cost": est_cost,
        }

    return {
        "agent": agent_name,
        "success": True,
        "provider": selected_provider.name,
        "model": selected_provider.model,
        "input_tokens": request.input_tokens,
        "output_tokens": simulated_output_tokens,
        "estimated_cost": est_cost,
        "status": execution_status,
    }


def run_demo() -> Dict[str, Any]:
    """Run full multi-agent simulation with routing, failover, and telemetry."""
    print("=" * 70)
    print("  OpenAIOps: Multi-Agent Routing, Telemetry & Failover Demo")
    print("=" * 70)

    # Step 1: Initialize OpenTelemetry with an in-memory exporter
    exporter = InMemorySpanExporter()
    init_telemetry(service_name="open-aiops-demo", exporter=exporter, force_reset=True)
    router = create_demo_router()
    print("\n[1] Initialized OpenTelemetry tracing and ModelRouter with 3 providers:")
    print("    - cloud-primary (gpt-4o, fast 350ms, $0.002 in / $0.006 out)")
    print("    - cloud-fallback (claude-3-5-sonnet, 500ms, $0.004 in / $0.010 out)")
    print("    - local-economy (llama-3-70b-instruct, 900ms, $0.0005 in / $0.0008 out)\n")

    results: List[Dict[str, Any]] = []

    # Step 2: Agent A (Data Ingestion & Summarizer - Cost-sensitive background task)
    print("[2] Executing Agent A (Data Ingestion - Budget-conscious)...")
    req_a = RouteRequest(
        input_tokens=2500,
        max_output_tokens=500,
        max_cost=0.01,
        max_latency_ms=1000.0,
    )
    res_a = simulate_agent_step(router, "data_summarizer", req_a, simulated_output_tokens=320)
    results.append(res_a)
    print(
        f"    -> Routed to: {res_a['provider']} ({res_a['model']}) | Cost: ${res_a['estimated_cost']:.5f}"
    )

    # Step 3: Agent B (Code Reviewer - Latency-sensitive interactive task)
    print("\n[3] Executing Agent B (Code Reviewer - Strict latency budget < 400ms)...")
    req_b = RouteRequest(
        input_tokens=1200,
        max_output_tokens=800,
        max_latency_ms=400.0,
    )
    res_b = simulate_agent_step(router, "code_reviewer", req_b, simulated_output_tokens=650)
    results.append(res_b)
    print(
        f"    -> Routed to: {res_b['provider']} ({res_b['model']}) | Cost: ${res_b['estimated_cost']:.5f}"
    )

    # Step 4: Simulate Outage / Overload on Primary Provider and Automated Failover
    print("\n[4] Simulating Rate-Limit Spike (HTTP 429) & Automated Failover...")
    req_c = RouteRequest(
        input_tokens=800,
        max_output_tokens=400,
        max_latency_ms=600.0,
    )
    # 4a: First call hits rate limit on cloud-primary
    print("    -> Agent C attempting call to cloud-primary...")
    res_c_fail = simulate_agent_step(
        router, "assistant", req_c, simulated_output_tokens=0, fail_provider=True
    )
    results.append(res_c_fail)
    print(f"    -> [ERROR CAUGHT] {res_c_fail['error']}")
    print("    -> Telemetry span recorded error state with exception.")

    # 4b: Governance layer marks provider overloaded and retries
    print("    -> Marking 'cloud-primary' as overloaded...")
    router.mark_overloaded("cloud-primary")

    print("    -> Retrying request for Agent C...")
    res_c_retry = simulate_agent_step(
        router, "assistant", req_c, simulated_output_tokens=280, fail_provider=False
    )
    results.append(res_c_retry)
    print(
        f"    -> Failover successfully routed to: {res_c_retry['provider']} ({res_c_retry['model']}) | Cost: ${res_c_retry['estimated_cost']:.5f}"
    )

    # Step 5: Provider Recovery
    print("\n[5] Simulating Provider Recovery...")
    print("    -> Health check: 'cloud-primary' restored to healthy state.")
    router.mark_healthy("cloud-primary")
    res_recovered = simulate_agent_step(router, "assistant", req_c, simulated_output_tokens=150)
    results.append(res_recovered)
    print(
        f"    -> Post-recovery request automatically returned to: {res_recovered['provider']} ({res_recovered['model']})"
    )

    # Step 6: Telemetry Spans Verification
    spans = exporter.get_finished_spans()
    print("\n" + "=" * 70)
    print(f"  Telemetry Verification: {len(spans)} Spans Captured via OpenTelemetry")
    print("=" * 70)
    total_tokens = 0
    for idx, span in enumerate(spans, 1):
        attrs = span.attributes or {}
        span_tokens = attrs.get("gen_ai.usage.total_tokens", 0)
        total_tokens += span_tokens
        model = attrs.get("gen_ai.request.model", "n/a")
        agent = attrs.get("agent.name", "unknown")
        status_name = "OK" if span.status.status_code == StatusCode.OK else "ERROR"
        print(
            f"  Span #{idx}: {span.name:<25} | Status: {status_name:<5} | Agent: {agent:<15} | Model: {model:<22} | Tokens: {span_tokens}"
        )

    print("-" * 70)
    print(f"  Total Processed Tokens: {total_tokens}")
    print("  Demo completed successfully!\n")

    return {
        "status": "success",
        "requests_processed": len(results),
        "spans_recorded": len(spans),
        "total_tokens": total_tokens,
    }


if __name__ == "__main__":
    run_demo()
