"""SafeFlow rewrite stage for T5-flagged responses.

Instead of blocking every response the ToxicChat T5 moderation filter flags,
this package routes the flagged (prompt, response) pair with an open-source
probe and rewrites the response with a small local model:

1. **Routing** (:mod:`rewriter.routing`): a linear-probe router over frozen
   Qwen3-4B activations (the public ``safeflow-routing-probe`` package)
   produces a REFUSE/REWRITE decision and a content domain per prompt.
   Deterministic, one forward pass, no API call.
2. **Prompt selection** (:mod:`rewriter.prompts`): GEPA-optimized rewrite
   prompts (960-metric-call budget, optimized directly against the T5
   filter), one per dense domain plus a unified fallback and a dedicated
   refusal prompt.
3. **Rewrite** (:mod:`rewriter.rewrite`): Qwen/Qwen3-4B-Instruct-2507
   rewrites the flagged response (greedy decoding, temperature 0); REFUSE
   rows get a contextual refusal instead.

End-to-end effect on the published 5,654-row GPT-5 evaluation set, under the
response-filter scenario (block iff the final response is T5-flagged), on the
230 T5-flagged rows: usefulness 78.3% -> 84.3%, harmful-responses-shown
5 -> 0, block rate ~flat (13.0% -> 12.2%).

See ``rewriter/README.md`` for usage and reproduction instructions.
"""

__version__ = "1.0.0"
