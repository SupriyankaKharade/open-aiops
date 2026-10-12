"""Semantic convention constants for OpenAIOps observability (INT-18).

Aligns OpenAIOps spans and metrics with OpenTelemetry GenAI semantic conventions
and internal operational attributes.
"""

from __future__ import annotations

# OpenTelemetry GenAI Semantic Conventions
ATTR_GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
ATTR_GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
ATTR_GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
ATTR_GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
ATTR_GEN_AI_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"
ATTR_GEN_AI_ATTEMPT_NUMBER = "gen_ai.attempt.number"

# Error Conventions
ATTR_ERROR_TYPE = "error.type"

# OpenAIOps Domain Attributes
ATTR_OPENAIOPS_REQUEST_ID = "openaiops.request_id"
ATTR_OPENAIOPS_ATTEMPT_ID = "openaiops.attempt_id"
ATTR_OPENAIOPS_TENANT_ID = "openaiops.tenant_id"
ATTR_OPENAIOPS_LATENCY_S = "openaiops.latency_s"
ATTR_OPENAIOPS_COST_TOTAL = "openaiops.cost.total"
ATTR_OPENAIOPS_COST_STATUS = "openaiops.cost.status"

# Event Names for Span Milestones
EVENT_ROUTING_DECIDED = "gen_ai.routing.decided"
ATTR_ROUTING_RANKED = "gen_ai.routing.ranked"
ATTR_ROUTING_REJECTED = "gen_ai.routing.rejected"

EVENT_RETRY_SCHEDULED = "gen_ai.retry.scheduled"
ATTR_RETRY_DELAY_S = "gen_ai.retry.delay_s"
ATTR_RETRY_ERROR_TYPE = "gen_ai.retry.error_type"

EVENT_FALLBACK_TRIGGERED = "gen_ai.fallback.triggered"
ATTR_FALLBACK_FROM = "gen_ai.fallback.from_provider"
ATTR_FALLBACK_TO = "gen_ai.fallback.to_provider"
ATTR_FALLBACK_ERROR_TYPE = "gen_ai.fallback.error_type"

EVENT_STREAM_FIRST_CHUNK = "gen_ai.stream.first_chunk"
