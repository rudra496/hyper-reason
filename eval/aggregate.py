"""Aggregate a raw eval JSONL into honest headline numbers + a labeled table.

Never re-derives numbers from the model — only reads what gsm8k_mini.py recorded.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict


def aggregate(path: str):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if not rows:
        return None
    methods = sorted({m for r in rows for m in r.get("methods", {})})
    n = len(rows)
    backend = rows[0].get("backend", "?")

    stats = {}
    for m in methods:
        correct = sum(1 for r in rows if r["methods"][m].get("correct"))
        unpars = sum(1 for r in rows if r["methods"][m].get("unparsable"))
        lat = [r["methods"][m].get("latency_ms", 0) for r in rows if "latency_ms" in r["methods"][m]]
        ctoks = [r["methods"][m].get("ctoks", 0) for r in rows if "ctoks" in r["methods"][m]]
        ptoks = [r["methods"][m].get("ptoks", 0) for r in rows if "ptoks" in r["methods"][m]]
        flash = [r["methods"][m].get("flashkv_saved_tokens", 0) or 0 for r in rows]
        stats[m] = {
            "accuracy_pct": round(100.0 * correct / n, 1),
            "correct": correct, "n": n,
            "unparsable_pct": round(100.0 * unpars / n, 1),
            "mean_completion_tokens": round(sum(ctoks) / max(1, len(ctoks)), 1),
            "mean_prompt_tokens": round(sum(ptoks) / max(1, len(ptoks)), 1),
            "mean_latency_ms": round(sum(lat) / max(1, len(lat)), 1),
            "mean_flashkv_saved_tokens": round(sum(flash) / max(1, len(flash)), 1),
        }
    return {"n": n, "backend": backend, "methods": stats}


def render_table(agg):
    lines = []
    lines.append(f"# GSM8K-mini results (N={agg['n']}, backend={agg['backend']})\n")
    lines.append("Honesty note: every number below is computed from the raw per-problem JSONL. "
                 "AE-MCTS entropy = sample-diversity proxy (no logprobs). VRAM = projected simulator "
                 "(no real GPU). Not a claim of SOTA.\n")
    lines.append("| method | accuracy | unparsable | mean compl. tokens | mean latency (ms) |")
    lines.append("|---|---:|---:|---:|---:|")
    for m, s in agg["methods"].items():
        lines.append(
            f"| {m} | {s['accuracy_pct']}% ({s['correct']}/{s['n']}) | {s['unparsable_pct']}% | "
            f"{s['mean_completion_tokens']} | {s['mean_latency_ms']} |"
        )
    mcts = agg["methods"].get("mcts", {})
    if mcts:
        lines.append(f"\nAE-MCTS mean projected FlashKV saved tokens/problem: "
                     f"{mcts.get('mean_flashkv_saved_tokens', 0)} (simulator, no real GPU).")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("usage: python eval/aggregate.py <runs.jsonl>")
        sys.exit(1)
    path = sys.argv[1]
    agg = aggregate(path)
    if not agg:
        print("no rows"); sys.exit(1)
    table = render_table(agg)
    print(table)
    base = os.path.splitext(path)[0]
    with open(base + ".summary.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    with open(base + ".summary.md", "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(f"\nwrote {base}.summary.json and {base}.summary.md")


if __name__ == "__main__":
    main()
