# Eval Changelog

Pre-registration is `CONFIG.md`. Changes before/after runs are logged here with justification.

## 2026-08-10 — pre-execution amendment (before any problem ran)
- N: 50 → **20** (first 20 of test).
- AE-MCTS: sims 24 → **6**, k 4 → **2**, depth 4 → **3**, max_tokens_per_step 128 → **160**.
- **Justification:** feasibility. The Z.AI gateway is stateless (no cross-sibling KV reuse),
  so AE-MCTS cost = `sims × k` model calls per problem. The original budget (24×4=96/problem →
  ~4800 calls over 50 problems) would take hours and offer no scientific advantage for a
  headline honest number. The reduced budget (6×2=12/problem → ~240 calls) runs in minutes and
  is reproducible. Larger runs remain possible via `python eval/gsm8k_mini.py --n 100 --sims 12`.
- SC-no-tree K stays 4; greedy T=0, 1 sample. These are unchanged.
