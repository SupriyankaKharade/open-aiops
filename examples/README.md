# Open-AIOps Examples & Demos

This directory contains executable scenarios and demonstrations of the Open-AIOps control plane.

## End-to-End Multi-Agent & Failover Demo (`run_demo.py`)

Simulates multi-agent requests and the **V1 Golden Scenario** per `docs/architecture.html` (Sections 4.2 & 7.2):

1. **Cost-Optimized Background Agent (`data_summarizer`)**: Evaluates provider costs and routes to the most economical option (`local-economy`).
2. **Latency-Sensitive Interactive Agent (`code_reviewer`)**: Enforces strict latency budgets (< 400ms) and routes to low-latency models (`cloud-primary`).
3. **Automated Failover on HTTP 429 (`chat_assistant`)**: Simulates a rate-limit spike on the primary model, executing transparent failover to `cloud-fallback` within remaining request deadlines.
4. **OpenTelemetry Telemetry Verification**: Inspects exported spans for token counts, latencies, and agent names.
5. **Architectural Invariant Audits**: Verifies that no API keys or raw prompts leak into spans, and confirms that telemetry failures never interrupt core execution.

### Running the Demo

```bash
python examples/run_demo.py
```
