# Response Generation

Generate model responses for `user_input` prompts in a JSONL dataset using Azure OpenAI.

## Setup

```bash
# Activate the virtualenv with openai + azure-identity installed
source ~/.virtualenvs/openai/bin/activate
```

## Authentication

The script supports two auth methods (checked in order):

1. **API Key** (preferred) — set the env vars:
   ```bash
   export AZURE_OPENAI_API_KEY="<your-key>"
   export AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com/"
   ```

2. **Identity-based** — uses `DefaultAzureCredential` (requires `az login` or managed identity). Set `AZURE_OPENAI_ENDPOINT` to your Azure OpenAI resource endpoint.

## Usage

```bash
cd ~/mod-frontier

python generation/generate_responses.py \
  -i data/toxicchat_with_relevance.jsonl \
  -o data/toxicchat_with_relevance.jsonl \
  -m gpt-5 \
  -w 16
```

### Arguments

| Flag | Description |
|------|-------------|
| `-i`, `--input` | Input JSONL file (must have `user_input` field) |
| `-o`, `--output` | Output JSONL file (default: same as input) |
| `-m`, `--model` | Model to use (configured in `MODEL_CONFIGS` dict) |
| `-w`, `--workers` | Number of concurrent requests (default: 16) |

## Adding a New Model

Edit `MODEL_CONFIGS` in `generate_responses.py`:

```python
MODEL_CONFIGS = {
    "gpt-5": {
        "endpoint": AZURE_OPENAI_ENDPOINT,
        "deployment": "gpt-5",
        "api_version": "2024-12-01-preview",
        "output_field": "model_output_gpt5",
    },
    "gpt-4o": {
        "endpoint": AZURE_OPENAI_ENDPOINT,
        "deployment": "gpt-4o",
        "api_version": "2024-12-01-preview",
        "output_field": "model_output_gpt4o",
    },
}
```

Then run with `-m gpt-4o`.

## Features

- **Resumable**: Skips rows that already have the output field populated. Safe to re-run after interruption.
- **Atomic writes**: Output is written to a temp file first, then moved into place — no partial writes.
- **Retry with backoff**: Retries on 429 / 5xx / timeout errors with exponential backoff (up to 6 retries).
- **Reasoning model detection**: Automatically omits `temperature` for o-series and gpt-5 models.
