"""Semantic convention constants for OpenAIOps observability (INT-18).

Aligns OpenAIOps spans and metrics with OpenTelemetry GenAI semantic conventions
and internal operational attributes.
"""

from __future__ import annotations

# OpenTelemetry GenAI Standard Semantic Conventions
ATTR_GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
ATTR_GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
ATTR_GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
ATTR_GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
ATTR_GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

# Standard Error Conventions
ATTR_ERROR_TYPE = "error.type"

# OpenAIOps Domain Attributes (Non-standard GenAI attributes live under openaiops.*)
ATTR_OPENAIOPS_REQUEST_ID = "openaiops.request_id"
ATTR_OPENAIOPS_ATTEMPT_ID = "openaiops.attempt_id"
ATTR_OPENAIOPS_ATTEMPT_NUMBER = "openaiops.attempt.number"
ATTR_OPENAIOPS_USAGE_TOTAL_TOKENS = "openaiops.usage.total_tokens"
ATTR_OPENAIOPS_TENANT_ID = "openaiops.tenant_id"
ATTR_OPENAIOPS_LATENCY_S = "openaiops.latency_s"
ATTR_OPENAIOPS_COST_TOTAL = "openaiops.cost.total"
ATTR_OPENAIOPS_COST_STATUS = "openaiops.cost.status"

# Milestones & Event Names under openaiops.*
EVENT_ROUTING_DECIDED = "openaiops.routing.decided"
ATTR_ROUTING_RANKED = "openaiops.routing.ranked"
ATTR_ROUTING_REJECTED = "openaiops.routing.rejected"

EVENT_RETRY_SCHEDULED = "openaiops.retry.scheduled"
ATTR_RETRY_DELAY_S = "openaiops.retry.delay_s"
ATTR_RETRY_ERROR_TYPE = "openaiops.retry.error_type"

EVENT_FALLBACK_TRIGGERED = "openaiops.fallback.triggered"
ATTR_FALLBACK_FROM = "openaiops.fallback.from_provider"
ATTR_FALLBACK_TO = "openaiops.fallback.to_provider"
ATTR_FALLBACK_ERROR_TYPE = "openaiops.fallback.error_type"

EVENT_STREAM_FIRST_CHUNK = "openaiops.stream.first_chunk"
ATTR_STREAM_TS = "openaiops.stream.ts"

EVENT_CIRCUIT_STATE_CHANGED = "openaiops.circuit.state_changed"
ATTR_CIRCUIT_PROVIDER = "openaiops.circuit.provider"
ATTR_CIRCUIT_FROM_STATE = "openaiops.circuit.from_state"
ATTR_CIRCUIT_TO_STATE = "openaiops.circuit.to_state"

# Backward compatibility aliases
ATTR_GEN_AI_ATTEMPT_NUMBER = ATTR_OPENAIOPS_ATTEMPT_NUMBER
ATTR_GEN_AI_USAGE_TOTAL_TOKENS = ATTR_OPENAIOPS_USAGE_TOTAL_TOKENS
