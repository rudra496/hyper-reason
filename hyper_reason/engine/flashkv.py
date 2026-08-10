"""FlashKV — HONEST projected KV-cache memory accounting (a simulator, not a measurement).

v1.x ``flash_kv.py`` held empty blocks (``key_data=None``) and reported "saved VRAM" from a
hardcoded ``0.032 MB/block`` constant. This module replaces that fiction with a transparent
simulator grounded in REAL per-node token counts (``TreeNode.kv_tokens``, computed from the
actual accumulated trajectory text).

What it computes (model-agnostic, in TOKENS and BLOCKS — both real):
  - **naive** = if every search node stored its OWN full KV (the standard-MCTS duplication).
  - **cow**   = Copy-on-Write paged sharing: each node stores only its DELTA over the parent's
                prefix; siblings share the parent's blocks by reference (refcount++).
  - savings = naive − cow.

It also emits a **projected** MB figure, but only behind an explicit ``bytes_per_token``
parameter (default is a clearly-labeled Llama-3-8B-class assumption). Every output dict carries
``"mode": "projected_simulator"`` and ``"no_real_gpu": True`` so it can never be mistaken for a
measured benchmark.
"""

from __future__ import annotations

import math
from typing import Any

# Documented default: Llama-3-8B-class KV footprint per token.
#   = 2 (K,V) * num_layers(32) * num_kv_heads(8) * head_dim(128) * 2 bytes(fp16) = 131_072 bytes
# This is an ASSUMPTION, disclosed in output. Set bytes_per_token for your model.
_DEFAULT_BYTES_PER_TOKEN = 2 * 32 * 8 * 128 * 2


class FlashKVSimulator:
    """Projects paged CoW KV-cache savings for a search tree from real token counts."""

    def __init__(self, block_size: int = 16, bytes_per_token: float = _DEFAULT_BYTES_PER_TOKEN):
        self.block_size = max(1, int(block_size))
        self.bytes_per_token = float(bytes_per_token)

    # -- the contract ReasonEngine.attach_flashkv/project expects -----------------
    def project(self, root: Any) -> dict:
        """Walk the tree, compute naive-vs-CoW block/token accounting, return projected stats."""
        if root is None:
            return self._empty()
        nodes = list(_walk(root))
        if not nodes:
            return self._empty()

        by_parent: dict[int, int] = {}  # id(node) -> its kv_tokens, for delta calc
        for n in nodes:
            by_parent[id(n)] = n.kv_tokens

        naive_tokens = 0
        cow_tokens = 0
        max_refcount = 1
        ref_counts: dict[int, int] = {}  # block-ish: count by prefix owner sharing
        leaves = 0

        for n in nodes:
            full = max(0, n.kv_tokens)
            naive_tokens += full  # naive: each node stores its full sequence
            parent_tok = by_parent.get(id(n.parent), 0) if getattr(n, "parent", None) is not None else 0
            delta = max(0, full - parent_tok)
            cow_tokens += delta  # CoW: only the new tokens this branch adds
            if getattr(n, "parent", None) is not None:
                # the parent's blocks gain one extra sharer
                pid = id(n.parent)
                ref_counts[pid] = ref_counts.get(pid, 0) + 1
            if not n.children:
                leaves += 1

        max_refcount = max([1] + list(ref_counts.values()))
        naive_blocks = _ceil_div(naive_tokens, self.block_size)
        cow_blocks = _ceil_div(cow_tokens, self.block_size)
        saved_tokens = naive_tokens - cow_tokens
        saved_pct = (saved_tokens / naive_tokens * 100.0) if naive_tokens else 0.0

        return {
            "mode": "projected_simulator",
            "no_real_gpu": True,
            "block_size_tokens": self.block_size,
            "nodes": len(nodes),
            "leaves": leaves,
            "naive_blocks": naive_blocks,
            "cow_shared_blocks": cow_blocks,
            "saved_tokens": saved_tokens,
            "saved_pct": round(saved_pct, 2),
            "max_block_sharing_refcount": max_refcount,
            "bytes_per_token_assumption": int(self.bytes_per_token),
            "projected_saved_mb": round(saved_tokens * self.bytes_per_token / (1024 * 1024), 3),
            "note": (
                "Projected CoW paged-sharing accounting from real per-node token counts. "
                "Not a GPU measurement. Set bytes_per_token for your model."
            ),
        }

    def _empty(self) -> dict:
        return {
            "mode": "projected_simulator", "no_real_gpu": True,
            "nodes": 0, "saved_tokens": 0, "saved_pct": 0.0,
            "note": "empty tree",
        }


def _walk(node):
    yield node
    for c in getattr(node, "children", []) or []:
        yield from _walk(c)


def _ceil_div(a: int, b: int) -> int:
    return math.ceil(a / b) if b else 0
