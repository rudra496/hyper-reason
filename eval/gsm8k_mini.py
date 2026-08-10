"""GSM8K-mini honest evaluation against a real model backend.

Runs three methods (greedy, self-consistency no-tree, AE-MCTS) on the first N problems of the
REAL GSM8K test split and writes RAW per-problem JSONL (anti-relapse: headline numbers come from
aggregate.py over this file, never re-derived). Config is pre-registered in CONFIG.md.

Usage:
  python eval/gsm8k_mini.py --n 20 --model glm-4.6
  python eval/gsm8k_mini.py --n 100 --sims 12   # larger reproducible run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import requests

# allow `python eval/gsm8k_mini.py` from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hyper_reason import ReasonEngine, SearchConfig
from hyper_reason.backends import GLMBackend, MockBackend
from hyper_reason.engine.flashkv import FlashKVSimulator
from hyper_reason.engine.verifier import self_consistency
from hyper_reason.engine.math_utils import extract_final_answer
from hyper_reason.engine.mcts import _PROMPT_TEMPLATE

GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/test.jsonl"
)
CACHE = os.path.join(os.path.dirname(__file__), ".gsm8k_test.jsonl")
RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")


def load_gsm8k(n: int):
    if not os.path.exists(CACHE):
        print(f"fetching GSM8K test split -> {CACHE}")
        r = requests.get(GSM8K_URL, timeout=30)
        r.raise_for_status()
        with open(CACHE, "w", encoding="utf-8") as f:
            f.write(r.text)
    out = []
    with open(CACHE, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            obj = json.loads(line)
            ref = parse_reference(obj["answer"])
            out.append({"idx": i, "question": obj["question"], "reference": ref})
    return out


def parse_reference(answer_field: str):
    # GSM8K final answer is after the last '#### '
    tail = answer_field.strip().split("####")[-1].strip().replace(",", "").replace("$", "")
    try:
        return float(tail)
    except ValueError:
        return None


def num(s: str):
    s = (s or "").strip().replace(",", "").replace("$", "")
    try:
        return float(s)
    except ValueError:
        return None


def is_correct(extracted: str, reference):
    if reference is None:
        return False
    v = num(extracted)
    return v is not None and abs(v - reference) < 1e-4


def make_backend(name: str, model: str):
    if name == "glm":
        return GLMBackend(model=model)
    if name == "mock":
        return MockBackend()
    raise SystemExit(f"unknown backend {name}")


def run_greedy(backend, question):
    t0 = time.monotonic()
    s = backend.sample(_PROMPT_TEMPLATE.format(problem=question), k=1, temperature=0.0,
                       max_tokens=400)[0]
    ans = extract_final_answer(s.text)
    return {
        "extracted_answer": ans, "ptoks": s.prompt_tokens, "ctoks": s.completion_tokens,
        "latency_ms": round((time.monotonic() - t0) * 1000, 1),
    }


def run_sc(backend, question, k=4):
    t0 = time.monotonic()
    samples = backend.sample(_PROMPT_TEMPLATE.format(problem=question), k=k,
                             temperature=0.7, max_tokens=400)
    traces = [s.text for s in samples]
    ans, conf, dist = self_consistency(traces)
    return {
        "extracted_answer": ans, "confidence": conf,
        "ptoks": sum(s.prompt_tokens for s in samples),
        "ctoks": sum(s.completion_tokens for s in samples),
        "latency_ms": round((time.monotonic() - t0) * 1000, 1),
    }


def run_mcts(backend, question, sims, k, depth):
    cfg = SearchConfig(num_simulations=sims, k_samples=k, max_depth=depth,
                       temperature=0.7, max_tokens_per_step=160)
    eng = ReasonEngine(backend, cfg)
    eng.attach_flashkv(FlashKVSimulator())
    res = eng.reason(question)
    m = res["metrics"]
    return {
        "extracted_answer": res["boxed_answer"], "confidence": res["confidence"],
        "ptoks": m["total_prompt_tokens"], "ctoks": m["total_completion_tokens"],
        "latency_ms": round(m["elapsed_seconds"] * 1000, 1),
        "model_calls": m["model_calls"], "depth_reached": m["max_depth_reached"],
        "flashkv_saved_tokens": (m.get("flashkv") or {}).get("saved_tokens", 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--backend", default="glm", choices=["glm", "mock"])
    ap.add_argument("--model", default="glm-4.6")
    ap.add_argument("--sims", type=int, default=6)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--methods", default="greedy,sc,mcts")
    args = ap.parse_args()

    os.makedirs(RUNS_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(RUNS_DIR, f"{ts}.jsonl")
    problems = load_gsm8k(args.n)
    backend = make_backend(args.backend, args.model)
    methods = args.methods.split(",")

    print(f"backend={backend.name} is_live={backend.is_live} | N={args.n} | methods={methods}")
    print(f"writing -> {out_path}")

    with open(out_path, "w", encoding="utf-8") as out:
        for p in problems:
            row = {"idx": p["idx"], "question": p["question"], "reference": p["reference"],
                   "backend": backend.name, "methods": {}}
            for meth in methods:
                try:
                    if meth == "greedy":
                        r = run_greedy(backend, p["question"])
                    elif meth == "sc":
                        r = run_sc(backend, p["question"])
                    elif meth == "mcts":
                        r = run_mcts(backend, p["question"], args.sims, args.k, args.depth)
                    else:
                        continue
                    r["correct"] = is_correct(r["extracted_answer"], p["reference"])
                    r["unparsable"] = r["extracted_answer"] == "__unparsable__"
                except Exception as e:  # one failure must not kill the run
                    r = {"error": repr(e)[:200], "correct": False, "unparsable": True}
                row["methods"][meth] = r
            out.write(json.dumps(row) + "\n")
            out.flush()
            marks = {m: ("✓" if row["methods"][m].get("correct") else "✗") for m in methods}
            print(f"  [{p['idx']:>3}/{args.n}] ref={p['reference']} {marks}")

    print(f"\ndone. raw -> {out_path}")
    print(f"aggregate: python eval/aggregate.py {out_path}")


if __name__ == "__main__":
    main()
