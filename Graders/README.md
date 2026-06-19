# Graders

LLM-as-judge graders used in the mod-frontier paper. Two graders ship here,
built on **one shared engine** so there is no duplicated calling/caching code:

| Grader spec        | Task            | Label space | Input        | Judge model(s) |
|--------------------|-----------------|-------------|--------------|----------------|
| `toxicity_v10`     | ToxicChat prompt safety | `0` / `1` (binary) | single text  | 3-way ensemble: `gpt-4.1` ×2 + `gpt-4.1-mini` |
| `relevance_v01`    | Response relevance      | `1` / `2` / `3`    | response **vs** prompt (paired) | `gpt-4o` |

* **`toxicity_v10`** is the final ToxicChat grader (held-out test **F1 = 0.870**):
  a majority vote of `gpt-4.1` (v07 prompt) + `gpt-4.1` (v02 prompt) +
  `gpt-4.1-mini` (v07 prompt).
* **`relevance_v01`** scores how well an assistant response addresses the user
  prompt on a refusal-aware 1–3 scale.

## Quickstart — grade a new model's responses

You ran a new model and now have a JSONL file where each line carries at least
the user prompt and that model's reply:

```json
{"user_input": "how do I reverse a list in python?", "model_output": "<your model's reply>"}
```

**0. One-time setup** (AAD auth — no API keys):

```bash
az login
source ~/.virtualenvs/openai/bin/activate
cd mod-frontier/Graders
```

**1. Relevance of the response** — adds `relevance_score` ∈ {1,2,3}
(does the reply address the prompt? refusal-aware):

```bash
python -m graders grade -g relevance_v01 \
    -f model_output -pf user_input \
    -i ../data/my_responses.jsonl \
    -o ../data/my_responses.jsonl        # safe to write back in place
```

**2. Prompt toxicity** — adds `toxicity_label` ∈ {0,1} (ToxicChat ensemble):

```bash
python -m graders grade -g toxicity_v10 \
    -f user_input \
    -i ../data/my_responses.jsonl \
    -o ../data/my_responses.jsonl
```

Each command **preserves every existing field and appends one new column**, so
you can chain them (feed each output back in). After both, every row has
`relevance_score` and `toxicity_label`.

> **Which text each grader judges**
> * `relevance_v01` compares the **response** (`-f model_output`) against the
>   **prompt** (`-pf user_input`) — this is what reflects your *new* model.
> * `toxicity_v10` classifies the **user prompt** (`-f user_input`) under the
>   ToxicChat policy. It depends only on the prompt, so it is **identical no
>   matter which model produced the response.** To score whether the
>   **response itself** is toxic, see *Grading response toxicity* below.

### Toxic content & speed

* The default endpoint (`config/endpoint.yaml`) is the **no-content-filter**
  Azure resource, so explicit prompts grade fine. To force a specific no-filter
  `gpt-4o` deployment for the (single-judge) relevance grader, set env vars:

  ```bash
  GRADERS_AZURE_ENDPOINT="https://<your-resource>.openai.azure.com" \
  GRADERS_DEPLOYMENT="gpt-4o-nofilter" \
  python -m graders grade -g relevance_v01 -f model_output -pf user_input \
      -i ../data/my_responses.jsonl -o ../data/my_responses.jsonl
  ```

  (Don't set `GRADERS_DEPLOYMENT` for `toxicity_v10` — it would collapse the
  3-model ensemble onto one deployment.)
* **Large files:** shard across several Azure resources in parallel with
  `../run_relevance_multi.sh` (edit its input path / endpoint list).
* Re-run a row from scratch (ignore the cache) with `--regrade`.

### Grading response toxicity (optional)

`toxicity_v10` is a *prompt* classifier. To approximate the **response's**
toxicity you can point it at the reply (`-f model_output`) — but its prompt is
written for user messages, so treat that as approximate — or add a dedicated
response-safety spec under `specs/` (see *Adding a grader*).

## Layout

```
Graders/
  config/endpoint.yaml      # Azure endpoint + model->deployment map (NO secrets; AAD auth)
  graders/
    azure_client.py         # shared async Azure OpenAI client (DefaultAzureCredential)
    core.py                 # shared engine: render, cache, parse, ensemble, metrics
    cli.py                  # unified `grade` / `evaluate` CLI
    __main__.py             # `python -m graders ...`
  prompts/
    toxicity/{v02,v07}.txt  # toxicity ensemble prompts ({text} placeholder)
    relevance/v01.txt       # relevance prompt ({prompt} / {response} placeholders)
  specs/
    toxicity_v10.json       # ensemble spec
    relevance_v01.json      # single-judge spec
  cache/                    # sha256(model+prompt+sample) -> {raw, label}
```

A **grader spec** declares everything that differs between graders — models,
prompts, whether input is paired, the JSON output field, the valid label space,
and how members are aggregated — so the engine itself stays grader-agnostic.

## Auth

Authentication uses Azure AD via `DefaultAzureCredential` (run `az login`); the
endpoint URL lives in `config/endpoint.yaml`. No API keys are stored, so this
directory is safe to push to a public repo.

## Usage

```bash
# Score response relevance against the prompt, adding `relevance_score` to each row:
python -m graders grade -g relevance_v01 \
    -f model_output -pf user_input \
    -i ../data/toxicchat_with_v10_grader.jsonl \
    -o ../data/toxicchat_with_relevance.jsonl

# Label prompt toxicity, adding `toxicity_label`:
python -m graders grade -g toxicity_v10 -f user_input \
    -i ../data/toxicchat_human_annotated.jsonl \
    -o ../data/toxicchat_toxicity.jsonl

# Validate a grader against gold binary labels:
python -m graders evaluate -g toxicity_v10 -f user_input \
    --label-field toxicity -i ../data/splits/test.jsonl
```

`grade` preserves every original field and appends one label column
(`--label-name` overrides the spec default). `evaluate` reports F1 / precision /
recall / accuracy for binary tasks.

### Flags

| Flag | Meaning |
|------|---------|
| `-g/--grader`        | Spec name from `specs/` (e.g. `relevance_v01`). |
| `-i/--input`         | Input JSONL. |
| `-o/--output`        | Output JSONL (`grade`); optional metrics JSON (`evaluate`). |
| `-f/--field`         | Field with the text to grade (the response, for relevance). |
| `-pf/--prompt-field` | Reference field for paired graders (the prompt). |
| `-w/--workers`       | Concurrent workers (default 16). |
| `--label-name`       | Output column name (defaults to the spec's `label_name`). |
| `--label-field`      | (`evaluate`) JSONL field holding gold 0/1 labels. |
| `--regrade`          | Ignore the cache and re-call the model. |
| `--limit N`          | Only process the first N rows. |

## Adding a grader

Drop a prompt under `prompts/<task>/<name>.txt` (use `{text}` for single input
or `{prompt}`/`{response}` for paired input) and a spec under
`specs/<name>.json`. No engine changes are needed.
