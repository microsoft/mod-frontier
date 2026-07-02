# Rewrite stage: rewrite T5-flagged responses instead of blocking them

The published pipeline blocks a response when the ToxicChat T5 moderation
filter flags it. This directory adds the next stage: **route the flagged
(prompt, response) pair with an open-source probe and rewrite the response
with a small local model**, so the user gets a safe, useful answer instead of
a block.

On the published 5,654-row GPT-5 evaluation set
(`data/toxicchat_with_GPT5Response.jsonl`, 230 rows T5-flagged), scored with
this repository's own graders and metrics, under the response-filter scenario
(block iff the *final* response is still T5-flagged):

| Flagged set (n=230) | Block rate | Harmful shown | Usefulness |
|---|---|---|---|
| Response filter alone (block every flagged row) | 100% | 0/0 | 0.0% |
| **+ rewrite stage (this PR)** | **12.2% (28/230)** | **0/202** | **84.3%** [95% CI 79.6, 88.7] |

Full-universe effect (all 5,654 rows, response filter): usefulness
92.6% → 96.0%, blocked 230 → 28, harmful-shown 35 → 35 (all 35 are
pass-through rows the filter never flagged; the rewrite stage adds **zero**).

## How it works

1. **Routing** — [`routing.py`](routing.py). The
   [safeflow-routing-probe](https://github.com/goodfire-ai/safeflow-routing-probe)
   package (attention-pooling probe heads over frozen Qwen3-4B layer-18
   activations) maps each flagged prompt to a REFUSE/REWRITE decision
   (threshold 0.161) and one of 8 content domains. Deterministic, one GPU
   forward pass, no API call in the serving path.
2. **Prompt selection** — [`prompts/`](prompts/). GEPA-optimized rewrite
   prompts (960 metric calls per scope, optimized *directly against the T5
   filter* with usefulness and harmlessness terms — see
   [`repro/gepa/`](repro/gepa/)): one prompt per dense domain
   (creative_writing, information_seeking, other), a unified fallback for
   sparse domains, and a dedicated refusal prompt.
3. **Rewrite** — [`rewrite.py`](rewrite.py). `Qwen/Qwen3-4B-Instruct-2507`
   (greedy, temperature 0, 2048 new tokens) rewrites the flagged response
   under the selected system prompt; REFUSE rows get a contextual refusal
   (policy area + safe alternative), with a one-shot retry when the model
   collapses to a bare one-liner refusal.

### Why probe routing? (arm ablation, flagged set, response scenario)

| Arm (routing for decision + domain) | Usefulness [95% CI] | Block | Harmful shown | Routing cost |
|---|---|---|---|---|
| **probe + probe (this PR)** | **84.3%** [79.6, 88.7] | 12.2% | **0**/202 | 1 GPU forward pass |
| LLM intent + probe domain | 83.5% [78.7, 87.8] | 13.0% | 1/200 | 1 API call/request |
| LLM intent + LLM domain | 84.8% [80.0, 89.1] | 11.7% | 3/203 | 2 API calls/request |
| No routing (unified prompt, rewrite all) | 87.0% [82.6, 91.3] | 10.0% | 2/207 | none |
| probe + probe, pre-optimization prompts | 78.3% | 13.0% | 5/200 | 1 GPU forward pass |

Probe+probe is the only arm that adds zero harmful responses, is statistically
tied on usefulness with the LLM-routed arms (overlapping CIs), and keeps the
serving path free of LLM calls. The no-routing arm's higher headline is
inflated by refusal-credit: without a REFUSE branch it answers rows that
should be refused, and the refusal-aware relevance grader credits some of
those (its 2 harmful-shown come from exactly this failure). The last row is
the same architecture with the pre-optimization prompt pack: the 960-call GEPA
re-optimization is worth +6.1pp usefulness (paired bootstrap 95% CI
[+0.4, +11.7]) and takes harmful-shown 5 → 0.

## Install

```bash
pip install -r rewriter/requirements.txt
```

Python ≥= 3.10, one GPU (~16 GB for the two Qwen3-4B loads; the probe and the
rewriter share the backbone weights' HF cache). The pinned probe revision is
the exact code the numbers above were measured with. Grading needs either
Azure AD auth (`az login`, per [`../Graders/README.md`](../Graders/README.md))
or `GRADERS_AUTH=openai` + `OPENAI_API_KEY` (routes the same grader specs
through api.openai.com — see [`grader_transport.py`](grader_transport.py)).

## Run

```bash
# 1. Rewrite the T5-flagged rows (GPU; ~2 h for 230 rows on one H100)
python rewriter/pipeline.py \
    -i data/toxicchat_with_GPT5Response.jsonl \
    -o rewrites.jsonl --arm rw_probe_probe

# 2. Re-screen the rewrites with the T5 filter (GPU)
python rewriter/eval_e2e.py t5-rewrites --rewrites rewrites.jsonl -o rewrites_t5.json

# 3. Grade the rewrites (API)
python rewriter/eval_e2e.py grade -i data/toxicchat_with_GPT5Response.jsonl \
    --rewrites rewrites.jsonl -o rewrites_grades.jsonl

# 4. Merge into the evaluation file as columns
python rewriter/eval_e2e.py assemble -i data/toxicchat_with_GPT5Response.jsonl \
    --rewrites rewrites.jsonl --grades rewrites_grades.jsonl \
    --t5-rewrites rewrites_t5.json \
    --t5-prompts t5_prompts.json --grade-prompts grader_prompts.json \
    -o data/toxicchat_with_GPT5Response.jsonl
# (t5-prompts / grade-prompts are one-time, arm-independent prompt-level
#  columns; produce them with the same script's t5-prompts / grade-prompts
#  subcommands. The published file already carries them.)

# 5. Metrics — the published metrics script works directly on the columns
python metrics/calculate_metrics.py -i data/toxicchat_with_GPT5Response.jsonl \
    --response-field T5_model_output_rw_probe_probe \
    --harm-field grader_model_output_rw_probe_probe \
    --relevance-field relevance_score_rw_probe_probe
# flagged-subset tables + bootstrap CIs:
python rewriter/eval_e2e.py metrics -i data/toxicchat_with_GPT5Response.jsonl -o metrics.json
```

## Data columns added to `data/toxicchat_with_GPT5Response.jsonl`

Following the file's `<base_field>_<system>` naming convention
(`*_gpt5` → `*_rw_probe_probe`):

| Field | Rows | Meaning |
|---|---|---|
| `model_output_rw_probe_probe` | 230 flagged | The rewrite (or contextual refusal) text |
| `grader_model_output_rw_probe_probe` | all | Harm label of what the system shows (rewrite's grade on flagged rows, original response's on pass-through rows) |
| `relevance_score_rw_probe_probe` | all | Relevance 1–3, same pass-through semantics |
| `T5_model_output_rw_probe_probe` | all | T5 flag of what the system shows (a still-flagged rewrite stays blocked) |
| `T5_user_input` | all | T5 on the user prompt (prompt-filter scenarios) |
| `grader_user_input` | all | `toxicity_v10` on the user prompt (the FP-rate reference) |

`grader_user_input` is graded, not human-labeled: `toxicity_v10` applied to
`user_input` exactly as `Graders/README.md` prescribes. Judge realization
variance makes single rows flip occasionally (a regrade agreed with the
committed labels on 5,651/5,654 rows); FP-rate comparisons between arms use
the same committed column, so the comparison is unaffected.

`tests/` pins the headline numbers to the committed columns
(`pytest tests/` — no GPU needed).

## Reproducing the stage from scratch

Everything upstream of the shipped prompt packs lives in [`repro/`](repro/)
(`pip install -r rewriter/repro/requirements-repro.txt`):

1. **Training data** — [`repro/build_dataset.py`](repro/build_dataset.py):
   harvest ready-made toxic responses (BeaverTails, RealToxicityPrompts),
   keep the T5-flagged ones, route with the probe, golden-label
   (toxicity_v10 + claude-opus), and build group-disjoint per-domain splits.
   The training corpus is fully external to the ToxicChat evaluation data.
2. **Prompt optimization** — [`repro/gepa/run_gepa.py`](repro/gepa/run_gepa.py):
   DSPy-GEPA per scope against the T5-composite reward
   ([`repro/gepa/reward.py`](repro/gepa/reward.py)), 960 metric calls,
   seeds in [`repro/gepa/seeds/`](repro/gepa/seeds/). One scope ≈ a few
   GPU-hours + reflection-model API calls.
3. **End-to-end eval** — `pipeline.py` + `eval_e2e.py` as above.

Regenerated rewrites are not byte-identical (GPU batching nondeterminism even
at temperature 0), so reproduction targets the metric level: on a fixed
30-row sample we verify T5-pass and harmlessness land inside the committed
run's 95% CIs.
