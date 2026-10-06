# open-aiops

OpenAIOps is an open-source AI operations control plane designed to provide intelligent LLM routing, policy governance, configuration management, and operational observability across model providers.

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
   git clone https://github.com/VaibhavSoni24/open-aiops.git
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
| `APP_NAME` | `str` | `"open-aiops"` | Application identifier |
| `DEBUG` | `bool` | `false` | Enable verbose debugging |
| `LOG_LEVEL` | `str` | `"INFO"` | Logging severity (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) |
| `DEFAULT_TIMEOUT_MS` | `float` | `30000.0` | Default request timeout in milliseconds (`> 0`) |
| `PROVIDERS` | `json` | `[]` | JSON array of provider configurations |
| `API_KEYS` | `json` | `{}` | JSON map of provider names to API keys |

### Example Provider Configuration

In `.env` or the environment:

```env
PROVIDERS='[{"name":"gpt-4","model":"gpt-4-turbo","cost_per_1k_input_tokens":0.01,"cost_per_1k_output_tokens":0.03,"expected_latency_ms":250.0,"priority":1}]'
```

### Programmatic Access

```python
from open_aiops.core.config import get_settings

settings = get_settings()
print(settings.app_name)
print(settings.providers)
```

Sensitive keys in `api_keys` are automatically masked when converting settings to string representations for safe logging.

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
