# ToxicChat T5-Large Reproduction

Reproduce the ToxicChat T5-Large toxicity classifier results from
[ToxicChat: Unveiling Hidden Challenges of Toxicity Detection in Real-World User-AI Conversation](https://arxiv.org/abs/2310.17389) (EMNLP 2023).

## Paper-Reported Results (toxicchat0124 test set)

| Model | Precision | Recall | F1 | AUPRC |
|---|---|---|---|---|
| ToxicChat-T5-large | 0.7983 | 0.8475 | 0.8221 | 0.8850 |
| OpenAI Moderation (threshold=0.02) | 0.5476 | 0.6989 | 0.6141 | 0.6313 |

## Setup

### 1. Download Data
```bash
cd mod-frontier/moderation
python download_data.py
```

### 2. Submit AML Job
```bash
python submit_aml_job.py --split test --wait
```

This runs on your configured Azure ML GPU compute and workspace (set via the
`AML_*` environment variables; see the table below).

### 3. Evaluate Results (locally, after job completes)
```bash
python evaluate.py --predictions outputs/predictions_test.jsonl
```

## Reproducing the 3 filter scenarios

The paper compares applying the T5 filter on the prompt, the response, or both.
The scripts above cover the prompt-only test-set reproduction; the scripts below
produce the response and both scenarios on a JSONL with `user_input` and
`model_output` (e.g. `data/toxicchat_with_GPT5Response.jsonl`).

```bash
# Filter on response only — score one custom field (adds T5_<label> column)
python submit_field_job.py \
  --data ../data/toxicchat_with_GPT5Response.jsonl \
  --field model_output_gpt5 --label T5_model_output_gpt5 --wait

# Filter on prompt & response — score both fields in one job
#   (uses inference_both.py; see AML_PIPELINE.md for the submit snippet)
```

## Files

| File | Description |
|---|---|
| `download_data.py` | Downloads ToxicChat v0124 dataset to `data/` |
| `inference.py` | Runs T5-Large inference on `user_input` for the test split (runs on AML GPU) |
| `inference_field.py` | Runs T5-Large on any single JSONL field (e.g. `model_output_gpt5`) — response scenario |
| `inference_both.py` | Runs T5-Large on both `user_input` and `model_output` — prompt & response scenario |
| `evaluate.py` | Computes metrics vs paper, extracts human-label performance |
| `submit_aml_job.py` | Submits the `inference.py` test-set job to your GPU compute target |
| `submit_field_job.py` | Submits an `inference_field.py` job for a custom field |
| `conda.yml` | Conda environment for AML |
| `AML_PIPELINE.md` | Quick reference for the both-fields AML pipeline |

## AML Compute

Configure your Azure ML workspace via environment variables (placeholders are
used by default):

| Env var | Description |
|---|---|
| `AML_COMPUTE_NAME` | GPU compute target (e.g. a 1x V100 cluster) |
| `AML_WORKSPACE_NAME` | Azure ML workspace name |
| `AML_SUBSCRIPTION_ID` | Azure subscription id |
| `AML_RESOURCE_GROUP` | Azure resource group |
