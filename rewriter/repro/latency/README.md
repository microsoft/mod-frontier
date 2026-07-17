# Latency measurement harness

Released code behind the latency numbers: a controlled, serving-style
measurement of the rewrite stage and the T5 filter, on one GPU.

## Methodology

**Rewrite pipeline** ([`measure_rewrite.py`](measure_rewrite.py)) — the
conditional cost of the rewrite stage itself: probe routing plus rewrite
generation, timed *after* an original response already exists and has been
flagged by the response filter. It excludes original-response generation,
the initial response-filter call, any buffering, and the final moderation
re-screen that the paper's serving policy applies before a rewrite is shown:

- **Dedicated server.** Generation runs on a dedicated vLLM (or any
  OpenAI-compatible) server, so latency is not confounded with model load or
  co-tenant traffic. The routing probe loads once in the measuring process,
  as it would in a serving deployment.
- **Serial dispatch.** One request at a time against an idle server — this
  measures latency, not throughput.
- **Streaming.** Time-to-first-token (TTFT) is recorded from the request send
  to the first streamed content token — i.e. when the rewrite server emits
  its first token. Under the paper's full-response re-screen-before-display
  policy that token is not yet user-visible (the complete rewrite is
  re-screened before display), so TTFT here is a serving diagnostic, not a
  user-visible-latency claim. A stream-then-retract policy, where tokens are
  shown as they stream, is a different serving policy not evaluated here.
- **Warm-up.** 3 warm-up samples run first and are excluded, so CUDA/cache
  warm-up doesn't inflate the stats.
- **Fixed sample.** 3 + 30 rows drawn from the T5-flagged set with seed 42.
- **Measured path = shipped path.** Routing, prompt selection, and message
  construction are imported from the `rewriter` package itself; the
  bare-refusal retry counts toward generation time when it fires.

**T5 filter** ([`measure_t5.py`](measure_t5.py)) — the filter's amortized
batched cost per sample at batch 32, prompt side and response side, using the
exact `moderation/` inference recipe (fp32, `max_new_tokens=5`) that produced
the accuracy numbers. Warm-up batches excluded, CUDA-synchronized timing.

## Report median and P90 beside the mean

The rewrite-stage latency distribution is **bimodal**: refusals and short
rewrites stream a few sentences and finish in well under a second, while long
rewrites generate up to 2048 tokens. The mean therefore tracks the long tail,
not the typical request. Reference run (30 samples, dedicated vLLM on one
H100 80GB, bfloat16):

| Stage | Mean | Median | P90 |
|---|---|---|---|
| Probe routing | 0.076 s | 0.075 s | 0.116 s |
| Prompt selection | ~5 µs | ~5 µs | ~5 µs |
| Time to first token | 0.036 s | 0.029 s | 0.060 s |
| Generation (incl. retry) | 1.52 s | 0.38 s | 3.78 s |
| **Rewrite-stage total (`e2e_s`)** | **1.60 s** | **0.47 s** | **3.86 s** |

`e2e_s` = route + prompt select + complete rewrite generation. It is not
full conversation end-to-end latency, and not time to final display
eligibility (the final moderation re-screen is outside this harness; see
`measure_t5.py` for the filter's per-sample cost).

T5 filter, same hardware, batch 32 over all 5,654 rows: 8.5 ms/sample
(prompt side), 11.8 ms/sample (response side), amortized.

Note: the batched offline evaluator (`rewriter/pipeline.py` at chunk 8 with
HF `generate`) reports ~71 s/row of *batch wall-clock* per row — every row in
a batch is charged the whole batch's wall time, and batches are padded to the
longest member at 2048 max tokens. That number measures the evaluation job,
not serving latency; this harness measures serving.

## Run

```bash
# T5 filter timing (GPU; ~2 min for the full file, --limit 64 to smoke)
python rewriter/repro/latency/measure_t5.py -o t5_latency.json

# Rewrite pipeline timing (GPU; needs a server — vLLM shown; ~5 min)
vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8100 \
    --gpu-memory-utilization 0.55 --max-model-len 16384 &
python rewriter/repro/latency/measure_rewrite.py \
    --base-url http://localhost:8100/v1 -o rewrite_latency.json
# --smoke: 1 warm-up + 3 samples at max_tokens=64
```

`vllm` is not in `rewriter/requirements.txt` (it's a server-side dependency
with its own CUDA constraints); the measuring client only needs the packaged
requirements. `--gpu-memory-utilization 0.55` leaves room for the routing
probe's backbone on the same GPU.
