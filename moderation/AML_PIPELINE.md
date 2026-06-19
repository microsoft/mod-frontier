# T5 ToxicChat AML Pipeline — Quick Reference

Run `lmsys/toxicchat-t5-large-v1.0` on any JSONL to get toxicity predictions on `user_input` and `model_output`.

## Quick Steps

```bash
cd mod-frontier/moderation
source ~/.virtualenvs/openai/bin/activate

# 1. Copy your input JSONL (must have user_input and model_output fields)
cp /path/to/your/data.jsonl data/input_all.jsonl

# 2. Submit the both-fields job to AML (see the copy-paste script at the bottom
#    of this file; it runs inference_both.py, polls, and downloads the output)
```

## What it does

- Prefix: `"ToxicChat: "` + text → T5 outputs "positive" (toxic=1) or "negative" (toxic=0)
- Adds `T5_user_input` and `T5_model_output` (0/1) to each row
- Output at: `/tmp/aml_t5_results/artifacts/outputs/t5_both_predictions.jsonl`

## AML Config (proven working)

| Setting | Value |
|---------|-------|
| Compute | `<your-gpu-compute>` (1x V100 16GB) |
| Workspace | `<your-workspace>` |
| Subscription | `<your-subscription-id>` |
| Resource Group | `<your-resource-group>` |
| Environment | `toxicchat-t5-env-cu118:1` |
| Pip | `torch==2.1.2` (cu118), `transformers==4.36.2`, sentencepiece, protobuf |

## Performance

~5,654 rows × 2 fields ≈ 7 min on V100 (~37 rows/sec per field).

## Customizing fields

Edit `inference_both.py` lines 63-64 to score different fields:

```python
user_texts = [str(r.get("user_input", "")) for r in rows]
model_texts = [str(r.get("model_output", "")) for r in rows]
```

## Full copy-paste submit script

```python
from azure.ai.ml import MLClient, command
from azure.identity import DefaultAzureCredential
import time, os, shutil

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id='<your-subscription-id>',
    resource_group_name='<your-resource-group>',
    workspace_name='<your-workspace>',
)

cmd = (
    "pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118 && "
    "pip install 'transformers==4.36.2' 'datasets==2.16.1' scikit-learn sentencepiece protobuf huggingface-hub && "
    "python inference_both.py --input data/input_all.jsonl --output outputs/t5_both_predictions.jsonl --batch-size 32"
)

job = command(
    code='.',
    command=cmd,
    environment='toxicchat-t5-env-cu118:1',
    compute='<your-gpu-compute>',
    display_name='t5-both-fields',
    experiment_name='toxicchat-t5-reproduction',
)

returned_job = ml_client.jobs.create_or_update(job)
print(f'Job: {returned_job.name}')

while True:
    time.sleep(30)
    info = ml_client.jobs.get(returned_job.name)
    print(f'  {info.status}')
    if info.status in ('Completed', 'Failed', 'Canceled'):
        break

# Download
dl = '/tmp/aml_t5_results'
if os.path.exists(dl): shutil.rmtree(dl)
ml_client.jobs.download(returned_job.name, download_path=dl, all=True)
print(f'Output: {dl}/artifacts/outputs/t5_both_predictions.jsonl')
```
