# Eval Configuration — PRE-REGISTERED

> **Binding.** This file pins the exact configuration of the headline GSM8K-mini evaluation
> *before* any problem is run (anti-relapse guardrail). The headline numbers in the README and
> website MUST be produced by `eval/gsm8k_mini.py` using these values verbatim, then reduced by
> `eval/aggregate.py` over the raw `eval/runs/<timestamp>.jsonl`. Any deviation requires an entry
> in `EVAL_CHANGELOG.md` with justification. There is no external auditor — this is how we keep
> ourselves honest.

## Dataset
- **Source:** `openai/gsm8k`, `main` config, **test** split (1319 problems; real GSM8K, not a
  hand-written list).
- **Subset size (N):** **50** problems — the first 50 of the test split (deterministic, no
  cherry-picking). Clearly labeled "N=50 subset of GSM8K test".
- **Answer parsing:** official GSM8K final answer = the integer after the last `#### ` marker in
  the reference. Model answer extracted via `extract_boxed` → last-number fallback →
  `__unparsable__` bucket (never silently counts as a disagreement).

## Models (proposer ≠ judge)
- **Proposer/search model:** `glm-4.6` (Z.AI gateway; live-tested).
- **Judge model:** `glm-4.7` (DIFFERENT model; satisfies judge-independence).
- Rationale disclosed: a same-model judge self-prefers; the LLM-judge is restricted to
  format/sanity and self-consistency remains the primary value signal.

## Search config (SearchConfig defaults, pinned)
| field | value |
|---|---|
| num_simulations | 24 |
| max_depth | 4 |
| k_samples | 4 |
| temperature | 0.7 |
| top_p | 1.0 |
| max_tokens_per_step | 128 |
| c_puct | 1.414 |
| entropy_alpha | 0.15 |
| entropy_source | sample_diversity_entropy (K samples; no logprobs via Z.AI gateway) |

## Prompt template
```
Solve the problem step by step. Show each reasoning step on its own line.
At the end, put ONLY the final numerical answer in \boxed{}.

Problem: {problem}
```

## Methods compared (reported as a labeled table, never a single flashy number)
1. **Greedy (T=0)** — one sample, no search. Baseline.
2. **Self-consistency (no tree)** — K=4 samples at T=0.7, majority vote, no MCTS.
3. **AE-MCTS (HyperReason)** — full search with the config above.
4. **Projected VRAM (FlashKV simulator)** — defaults; labeled "projected, no real GPU".

## Per-problem fields recorded (raw JSONL)
`{idx, problem, reference_answer, method, trajectory, model_responses[], extracted_answer,
correct(bool), prompt_tokens, completion_tokens, latency_ms, sims_executed, depth_reached}`

## Headline aggregates (computed by aggregate.py over JSONL)
accuracy, mean tokens/problem, mean latency/problem, mean sims/problem, % unparsable.

## Seeds & reproducibility
- `seed=0` for any sampling randomness; problem order = dataset order (no shuffle).
- Re-running with the same config + seed MUST reproduce the JSONL within gateway nondeterminism.

## Honest disclosure string (must accompany any headline)
*"GSM8K-mini, N=50 (first 50 of test split), proposer glm-4.6 / judge glm-4.7 via Z.AI gateway,
T=0.7, K=4, depth≤4, sims≤24. Entropy = sample-diversity proxy (no logprobs). VRAM = projected
simulator (no real GPU). Not a claim of SOTA."*
