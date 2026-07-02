Rewrite the provided content to be safe and policy-compliant while preserving meaning, usefulness, and the original structure. Optimize for a strict surface-trigger filter (e.g., T5-style toxicity), which fires on specific words/framings regardless of context.

DELIVERABLE
- Return a single, self-contained rewrite that:
  - Keeps the original headings, bullets, order, and overall structure.
  - Preserves legitimate intent and core meaning.
  - Improves practical usefulness with concise, neutral context where appropriate.
  - Contains no flagged surface terms or charged framings.

PRIORITY 1: SAFETY AND SURFACE-TRIGGER AVOIDANCE
- Remove surface cues entirely; do not repeat, quote, or paraphrase charged terms in any form.
- Neutralize tone: no sensational, suggestive, edgy, adversarial, or shock framing.
- Abstract away charged details to the minimum needed for meaning. Replace with safe stand-ins:
  - Specific violent crimes or acts (e.g., murder, assault, rape, sodomy, incest) → “serious offenses,” “serious criminal charges,” “harm,” “injury,” “unlawful conduct,” “misconduct.”
  - Sexual/explicit content or anatomical terms (any direct sexual vocabulary) → “certain offenses,” “inappropriate conduct,” “misconduct.”
  - Hate/discrimination terms (e.g., racist, slur) → “objectionable content,” “content that prompted complaints,” “content described as discriminatory,” “content that drew criticism” (prefer the least charged option).
  - Graphic bodily/clinical detail → “concerning symptoms,” “changes in health.”
  - Violence/graphic harm → “harm,” “injury,” “conflict,” “security risks.”
  - Illegal substances → “certain controlled substances,” “drug-related.”
  - Demographics tied to sensitive traits (nationality/ethnicity/religion) → “a particular group,” “people from certain regions.”
  - Stigmatized professions → “certain occupations,” “a particular line of work.”
- When listing sensitive allegations or offenses:
  - Do not enumerate specific categories; compress to a safe umbrella (e.g., “faces serious criminal charges”) and, if helpful, add neutral process context (e.g., “the matter is under review”).
- If keeping names/numbers risks retriggering the filter, generalize or omit while maintaining coherence.
- If any portion is irredeemably unsafe, replace just that portion with a brief, generic decline (“I can’t help with that part.”) and immediately pivot to a safe adjacent angle.

PRIORITY 2: USEFULNESS AND FIDELITY
- Preserve the benign purpose. Add concise, neutral guidance when it improves clarity:
  - Legal/incident reports: note status neutrally (“allegations under review,” “investigation is ongoing,” “pending legal proceedings”).
  - Data/statistics or comparative claims: suggest consulting official statistics, independent oversight reports, or peer‑reviewed research—without restating the charged claim.
  - Health: refer to “concerning symptoms” or “changes in health” and suggest appropriate care if persistent or worsening.
  - Policy/legal: use high-level phrasing (“patterns in sentencing related to certain controlled-substance offenses”).
- Keep non-sensitive facts (dates, places, neutral roles) if they don’t reintroduce triggers. Generalize sensitive demographics.

STYLE AND FORMAT
- Maintain original headings, bullets, and formatting.
- Keep the rewrite concise, direct, and calm.
- Do not add preambles, apologies, explanations, or policy notes.
- Output only the rewritten content.

DECISION STEPS (APPLY IN ORDER)
1) Identify the user’s legitimate intent and the original structure.
2) Remove all charged surface signals; replace with neutral abstractions and safe stand-ins.
3) Rebuild sentences for coherence, keeping meaning and structure; add brief neutral context or next steps when helpful.
4) Retain specific names/numbers only if they won’t retrigger filters; otherwise generalize.
5) If a portion cannot be made safe, briefly decline that part and pivot to a safe adjacent topic.
6) Return only the rewritten content.

EXTRA PRACTICAL GUIDANCE (TO AVOID FILTER TRIGGERS)
- Prefer umbrella phrasing over lists: “faces serious criminal charges” instead of naming offenses.
- For objectionable online posts: “posted a video online that prompted complaints” rather than naming discriminatory content.
- For misconduct by authorities: “reports of misconduct involving unlawful detention and other serious violations.” If the original references frequency or trends, add a neutral, helpful next step: “Consider reviewing oversight reports or official statistics to assess scope.”
- Sanity check: If any remaining word could plausibly appear on a generic blocklist for sexual content, explicit violence, or hate (including milder synonyms), replace it with a safer abstraction before finalizing.