# open-aiops

OpenAIOps is an open-source AI operations control plane designed to provide intelligent LLM routing, policy governance, configuration management, and operational observability across model providers.

Architecture docs: https://supriyankakharade.github.io/open-aiops/

---

## Architecture Overview

The repository is organized into modular packages:

```
open_aiops/
├── core/             # Core configuration, settings management, and shared models
├── governance/       # Model routing engine, fallback policies, and circuit breaking
└── observability/    # Metrics collection and telemetry pipeline
tests/                # Test suites for each subsystem
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher

### Environment Setup

1. **Clone the repository and set up a virtual environment:**

   ```bash
   git clone https://github.com/SupriyankaKharade/open-aiops.git
   cd open-aiops
   python -m venv .venv
   ```

2. **Activate the virtual environment:**

   - **Windows:**
     ```powershell
     .venv\Scripts\activate
     ```
   - **Linux / macOS:**
     ```bash
     source .venv/bin/activate
     ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

---

## Configuration

Configuration is managed in `open_aiops.core.config` using `pydantic-settings`. Settings can be supplied through environment variables or a local `.env` file.

### Environment File Setup

Copy `.env.example` to create a local `.env`:

```bash
cp .env.example .env
```

### Configuration Variables

| Variable | Type | Default | Description |
|---|---|---|---|
| `OPEN_AIOPS_APP_NAME` | `str` | `"open-aiops"` | Application identifier |
| `OPEN_AIOPS_DEBUG` | `bool` | `false` | Enable verbose debugging |
| `OPEN_AIOPS_LOG_LEVEL` | `str` | `"INFO"` | Logging severity (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) |
| `OPEN_AIOPS_DEFAULT_TIMEOUT_MS` | `float` | `30000.0` | Default request timeout in milliseconds (`> 0`) |
| `OPEN_AIOPS_PROVIDERS` | `json` | `[]` | JSON array of provider configurations |
| `OPEN_AIOPS_API_KEYS` | `json` | `{}` | JSON map of provider names to SecretStr API keys |

### Example Provider Configuration

In `.env` or the environment:

```env
OPEN_AIOPS_PROVIDERS='[{"name":"gpt-4","model":"gpt-4-turbo","cost_per_1k_input_tokens":0.01,"cost_per_1k_output_tokens":0.03,"expected_latency_ms":250.0}]'
```

### Programmatic Access

```python
from open_aiops.core.config import get_settings

settings = get_settings()
print(settings.app_name)
print(settings.providers)
```

Sensitive keys in `api_keys` are typed as Pydantic `SecretStr`, preventing raw credential leakage in `repr()`, `str()`, or `model_dump_json()`.

---

## Observability Metrics

The `MetricsSubscriber` in `open_aiops.observability` listens to gateway lifecycle events and records metrics instruments through a lightweight `MetricsRecorder` facade.

### Registering the Metrics Subscriber

`MetricsSubscriber` consumes lifecycle events through `EventDispatcher`:

```python
from open_aiops.events import EventDispatcher
from open_aiops.observability import MetricsSubscriber

dispatcher = EventDispatcher()
dispatcher.subscribe(MetricsSubscriber())
```

### Emitted Metrics Instruments

The current implementation emits the following OpenTelemetry instruments:

| Instrument Name | Type | Unit | Description |
|---|---|---|---|
| `openaiops.request.duration` | Histogram | `s` | End-to-end duration of completed requests in seconds |
| `openaiops.requests` | Counter | `{request}` | Total count of completed requests by outcome |
| `openaiops.provider.errors` | Counter | `{error}` | Total count of provider errors by canonical error type |
| `openaiops.retries` | Counter | `{retry}` | Total count of request retry attempts per provider |
| `openaiops.fallbacks` | Counter | `{fallback}` | Total count of provider fallbacks triggered |
| `openaiops.tokens.input` | Histogram | `{token}` | Distribution of input prompt tokens per attempt |
| `openaiops.tokens.output` | Histogram | `{token}` | Distribution of output completion tokens per attempt |
| `openaiops.cost` | Counter | (dimensionless) | Total estimated monetary cost of operations |

### Attribute Dimensions & Privacy Safeguards

Metric attributes are strictly constrained to low-cardinality dimensions:
- `provider`: Provider identifier (e.g., `"openai"`, `"anthropic"`).
- `model`: Bounded model name. Model strings are sanitized, length-limited, and bounded by a cardinality cap (`max_distinct_models`, default 100) before overflowing to `"other"`. Strings containing control characters normalize to `"other"`.
- `outcome`: Request outcome (`"success"`, `"error"`, or `"cancelled"`).
- `error_type`: Canonical INT-10 provider error class names (`"ProviderError"`, `"RateLimitError"`, `"ProviderTimeout"`, `"ProviderUnavailable"`, `"AuthError"`, `"BadRequest"`, or `"unknown"`).

**Privacy & Security Invariant:** High-cardinality and sensitive data—such as request IDs, tenant IDs, API keys, bearer tokens, prompt text, response completions, and raw exception messages—are strictly prohibited and never emitted as metric labels. Request IDs are used solely for internal bounded in-flight tracking and deduplication.

### OpenTelemetry SDK Configuration

`MetricsSubscriber` and `MetricsRecorder` depend only on the OpenTelemetry API, not the SDK. They do not configure metric exporters, readers, or install a global SDK provider. An application must configure its OpenTelemetry SDK and exporter/reader if it wants exported data; without an SDK provider the current metrics facade safely behaves as a no-op.

### Testing Metrics

Run the offline metrics test suites:

```bash
python -m pytest -q tests/test_metrics.py tests/test_metrics_subscriber.py
```

---

## Running Tests

Run the full test suite using `pytest`:

```bash
python -m pytest -q
```

Run specific test modules:

```bash
python -m pytest -q tests/test_config.py
```
