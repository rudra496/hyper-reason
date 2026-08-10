<div align="center">

# ⚡ HyperReason

### A test-time-compute (AE-MCTS) engine that **actually runs** — and reports when it loses.

[![tests](https://github.com/rudra496/hyper-reason/actions/workflows/ci.yml/badge.svg)](https://github.com/rudra496/hyper-reason/actions/workflows/ci.yml)
[![Live demo](https://img.shields.io/badge/site-live%20demo-00e5ff)](https://rudra496.github.io/hyper-reason)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-blue)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**[🌐 Live in-browser demo](https://rudra496.github.io/hyper-reason)** · **[📊 Real eval (raw JSONL)](eval/runs/)** · **[🧾 Honest eval config](eval/CONFIG.md)**

</div>

> **v2 is a full honest rebuild.** v1.x shipped an engine that **never called a model** and a
> benchmark table that was invented. v2 actually drives a real LLM, and publishes the eval
> below — including the run where AE-MCTS **underperformed greedy**. No fabrication, no silent
> fake fallbacks, every number traceable to a command.

---

## Why this exists

Scaling test-time compute (o1/o3, DeepSeek-R1, rStar) works — but most public "MCTS for LLMs"
repos are demo theater: the wrapper never calls the model, the KV numbers are hardcoded, the
benchmark table is a wish. **HyperReason v2 is the boring, honest version**: a real
Adaptive-Entropy MCTS that genuinely samples a model, searches, votes by self-consistency, and
projects KV-cache savings from real token counts — with an eval you can re-run line by line.

## 1-line quickstart

```bash
pip install -e .            # or: pip install hyper-reason
export ANTHROPIC_BASE_URL=https://api.z.ai/api/anthropic   # any Anthropic-compatible gateway
export ANTHROPIC_API_KEY=...
```
```python
from hyper_reason import wrap_model, GLMBackend

model = wrap_model(GLMBackend(model="glm-4.6"))   # or OllamaBackend / TransformersBackend / None (Mock)
res = model.reason("Janet has 3 boxes of 12 apples, gives away 5 and eats 2. How many remain?")

print(res["boxed_answer"])          # the self-consistency answer
print(res["confidence"])            # agreement fraction over terminal trajectories
print(res["metrics"]["flashkv"])    # projected KV-cache savings (labeled: simulator, no real GPU)
```

```bash
# CLI
hyper-reason --problem "If a train travels 60mph for 3.5h, how far?" --backend glm
```

## Real benchmark — GSM8K-mini (N=20, GLM-4.6)

| method | accuracy | unparsable | mean tokens | mean latency |
|---|---:|---:|---:|---:|
| **greedy (T=0, 1 sample)** | **95.0%** (19/20) | 0% | 185 | 3.5s |
| self-consistency (K=4) | 90.0% (18/20) | 0% | 750 | 11.7s |
| AE-MCTS (sims=6, k=2, d≤3) | 85.0% (17/20) | 0% | 669 | 14.2s |

**The honest reading:** GLM-4.6 is already near the ceiling on easy GSM8K, so a *modest-budget*
search underperforms greedy here. Test-time search pays off on **harder problems / weaker base
models / bigger budgets** — not on a strong model at its ease. We publish the loss, not a fake
win. **Not a claim of SOTA.** Re-run / scale it: `python eval/gsm8k_mini.py --n 100 --sims 16 --k 4`. Full disclosure in
[`eval/CONFIG.md`](eval/CONFIG.md); raw per-problem traces in [`eval/runs/`](eval/runs/).

## What's real vs. labeled

| Component | Status |
|---|---|
| Model calls (GLM / Ollama / Transformers) | **Real** — backend actually sampled; `pip`-installed + live-tested on GLM |
| AE-MCTS search (sample → diversity entropy → AE-PUCT → self-consistency) | **Real** — bounded budget, model-generated candidates |
| Entropy | **Labeled**: `sample_diversity_entropy` — a logprob-free proxy (the gateway returns no logprobs) |
| KV-cache savings | **Labeled**: projected simulator from real per-node token counts (no real GPU) |
| Backends when offline | **Raise** — never fabricate a response (v1.x silently faked one) |

See the **[honesty contract](https://rudra496.github.io/hyper-reason#honesty)** and the JS↔Python parity test that pins the browser demo to the package.

## Architecture

```
hyper_reason/
  backends/      base(GLM/Ollama/Transformers/Mock) — the ONLY place a model is called
  engine/        mcts (AE-MCTS), entropy, verifier (self-consistency), flashkv (projected), math_utils
  orchestrator/  LangGraph proposer→verifier→refiner w/ checkpointer + human-in-the-loop interrupt
  wrapper.py     wrap_model() — the 1-line API
eval/            gsm8k_mini.py (real GSM8K) + aggregate.py + CONFIG.md (pre-registered) + runs/*.jsonl
docs/            single-file interactive site (pure-JS engine port, no third-party CDNs)
```

## For contributors

```bash
pip install -e ".[dev,orchestrator]"
pytest -q                          # 45 tests, incl. a live GLM end-to-end run
python eval/gsm8k_mini.py --n 20   # reproduce the headline numbers
```

Read [`eval/CONFIG.md`](eval/CONFIG.md) (pre-registered) and [`eval/EVAL_CHANGELOG.md`](eval/EVAL_CHANGELOG.md).
The honesty contract is binding: a fix isn't "done" until the eval re-runs and the raw JSONL reflects it.

## Citation

```bibtex
@software{sarker2026hyperreason,
  author = {Rudra Sarker},
  title  = {HyperReason v2: an honest Adaptive-Entropy MCTS test-time-compute engine},
  url    = {https://github.com/rudra496/hyper-reason},
  year   = {2026}
}
```

MIT licensed · built by [Rudra Sarker](https://github.com/rudra496) ·
[portfolio](https://rudra496.github.io/site)

> ⭐ If a test-time-search repo that *publishes its losses* is more useful than one that fakes
> its wins, star it and follow [@rudra496](https://github.com/rudra496).
