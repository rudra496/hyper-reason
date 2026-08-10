"""FlashKV simulator tests — honest, model-agnostic accounting from real token counts."""

import pytest

from hyper_reason.engine.flashkv import FlashKVSimulator
from hyper_reason.engine.mcts import TreeNode
from hyper_reason import ReasonEngine, MockBackend, SearchPresets


class _Fake:
    """Minimal node stand-in to prove project() only needs kv_tokens/children/parent."""


def test_empty_tree():
    out = FlashKVSimulator().project(None)
    assert out["nodes"] == 0 and out["saved_tokens"] == 0


def test_sharing_saves_tokens_and_is_labeled_simulated():
    # root + 2 children sharing the root prefix (siblings -> CoW wins)
    root = TreeNode("root prompt tokens " * 10)
    c1 = TreeNode("root prompt tokens " * 10 + " extra delta one", parent=root)
    c2 = TreeNode("root prompt tokens " * 10 + " extra delta two", parent=root)
    root.children = [c1, c2]
    out = FlashKVSimulator().project(root)
    assert out["mode"] == "projected_simulator"
    assert out["no_real_gpu"] is True
    assert out["saved_tokens"] > 0
    assert 0.0 <= out["saved_pct"] <= 100.0
    assert out["cow_shared_blocks"] <= out["naive_blocks"]
    assert out["max_block_sharing_refcount"] >= 2  # root shared by 2 children


def test_naive_ge_cow_always():
    root = TreeNode("a " * 20)
    out = FlashKVSimulator().project(root)
    assert out["naive_blocks"] >= out["cow_shared_blocks"]


def test_bytes_per_token_changes_mb_not_tokens():
    root = TreeNode("a " * 50)
    c = TreeNode("a " * 50 + " b " * 10, parent=root)
    root.children = [c]
    small = FlashKVSimulator(bytes_per_token=100).project(root)
    big = FlashKVSimulator(bytes_per_token=10000).project(root)
    assert small["saved_tokens"] == big["saved_tokens"]  # token count is model-agnostic
    assert big["projected_saved_mb"] > small["projected_saved_mb"]


def test_integrates_with_engine_via_attach():
    eng = ReasonEngine(MockBackend(), SearchPresets.ultra_fast())
    eng.attach_flashkv(FlashKVSimulator())
    res = eng.reason("If 4 workers build a wall in 6 hours, how long for 8 workers?")
    fk = res["metrics"]["flashkv"]
    assert fk["mode"] == "projected_simulator"
    assert fk["nodes"] > 0
    assert "Not a GPU measurement" in fk["note"]
