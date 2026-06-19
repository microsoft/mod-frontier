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
cd Goodfire_rewrite_paper/moderation
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

## Files

| File | Description |
|---|---|
| `download_data.py` | Downloads ToxicChat v0124 dataset to `data/` |
| `inference.py` | Runs T5-Large inference (runs on AML GPU) |
| `evaluate.py` | Computes metrics vs paper, extracts human-label performance |
| `submit_aml_job.py` | Submits AML job to your GPU compute target |
| `conda.yml` | Conda environment for AML |

## AML Compute

Configure your Azure ML workspace via environment variables (placeholders are
used by default):

| Env var | Description |
|---|---|
| `AML_COMPUTE_NAME` | GPU compute target (e.g. a 1x V100 cluster) |
| `AML_WORKSPACE_NAME` | Azure ML workspace name |
| `AML_SUBSCRIPTION_ID` | Azure subscription id |
| `AML_RESOURCE_GROUP` | Azure resource group |
