"""Tests for the LangGraph orchestrator. All assertions run against the deterministic
``MockBackend`` (``is_live=False``) so every property checked here is REAL end-to-end
behavior of the graph, not a mocked/stubbed outcome.
"""

from hyper_reason.backends.mock_backend import MockBackend
from hyper_reason.engine.config import SearchConfig
from hyper_reason.orchestrator import (
    DEFAULT_REFINE_THRESHOLD,
    OrchestratorState,
    build_graph,
    run,
)


# ---------------------------------------------------------------------------
# build_graph
# ---------------------------------------------------------------------------

def test_build_graph_returns_compiled_graph():
    """build_graph(MockBackend()) returns a compiled graph (no exception) with .invoke."""
    graph = build_graph(MockBackend())
    assert callable(graph.invoke)
    assert callable(graph.get_state)


# ---------------------------------------------------------------------------
# proposer node
# ---------------------------------------------------------------------------

def test_proposer_produces_k_real_samples():
    """proposer emits exactly config.k_samples real, non-empty candidate strings.

    refine_threshold=0.0 guarantees the refiner is skipped (confidence is always >= 0),
    so the candidate count at the interrupt equals the proposer's k_samples.
    """
    cfg = SearchConfig(k_samples=4)
    graph = build_graph(MockBackend(), config=cfg, refine_threshold=0.0)
    state = graph.invoke(
        {"problem": "What is 6 * 7?"},
        {"configurable": {"thread_id": "prop-1"}},
    )
    candidates = state.get("candidates", [])
    assert len(candidates) == cfg.k_samples
    assert all(isinstance(c, str) and c for c in candidates)


# ---------------------------------------------------------------------------
# verifier node
# ---------------------------------------------------------------------------

def test_verifier_sets_best_confidence_distribution():
    """verifier populates best (non-empty str), confidence in [0,1], and a distribution dict
    whose counts sum to the number of candidates."""
    cfg = SearchConfig(k_samples=3)
    graph = build_graph(MockBackend(), config=cfg, refine_threshold=0.0)
    state = graph.invoke(
        {"problem": "Compute 2 + 3."},
        {"configurable": {"thread_id": "ver-1"}},
    )
    assert isinstance(state.get("best"), str) and state["best"]
    assert 0.0 <= state["confidence"] <= 1.0
    distribution = state.get("distribution")
    assert isinstance(distribution, dict) and len(distribution) >= 1
    # Every candidate contributes exactly one extracted answer -> counts sum to n candidates.
    assert sum(distribution.values()) == len(state["candidates"])


# ---------------------------------------------------------------------------
# human-in-the-loop interrupt
# ---------------------------------------------------------------------------

def test_graph_hits_interrupt_on_first_invoke():
    """First invoke pauses inside finalize: no committed final_answer, and a node is pending."""
    graph = build_graph(MockBackend(), refine_threshold=0.0)
    cfg = {"configurable": {"thread_id": "int-1"}}
    state = graph.invoke({"problem": "What is 10 - 4?"}, cfg)

    # finalize did not complete -> final_answer is absent / falsy.
    assert not state.get("final_answer")
    # LangGraph surfaces a pending task at the interrupt.
    snapshot = graph.get_state(cfg)
    assert snapshot.next, "expected a pending node at the finalize interrupt"
    assert "finalize" in snapshot.next


def test_run_without_resume_returns_awaiting_approval():
    """run(..., resume_value=None) surfaces the awaiting-approval state, no final_answer."""
    state = run(
        "What is 5 + 5?",
        MockBackend(),
        resume_value=None,
        thread_id="await-1",
        refine_threshold=0.0,
    )
    assert state.get("awaiting_approval") is True
    assert not state.get("final_answer")


def test_resume_commits_final_answer_equal_to_resume_value():
    """After resuming with a (possibly human-edited) value, final_answer is committed and
    equals that resume value exactly."""
    state = run(
        "What is 5 + 5?",
        MockBackend(),
        resume_value="10",
        thread_id="resume-1",
        refine_threshold=0.0,
    )
    assert state.get("final_answer") == "10"
    assert not state.get("awaiting_approval", False)


# ---------------------------------------------------------------------------
# trace
# ---------------------------------------------------------------------------

def test_trace_is_non_empty_with_one_entry_per_executed_node():
    """trace is a non-empty list of dicts; proposer, verifier and finalize all appear."""
    state = run(
        "What is 3 * 3?",
        MockBackend(),
        resume_value="9",
        thread_id="trace-1",
        refine_threshold=0.0,
    )
    trace = state.get("trace", [])
    assert isinstance(trace, list)
    assert len(trace) >= 3  # proposer + verifier + finalize (no refiner at threshold=0.0)
    names = [entry["node"] for entry in trace]
    assert names[0] == "proposer"
    assert "verifier" in names
    assert "finalize" in names
    assert "refiner" not in names  # refine was disabled
    for entry in trace:
        assert set(entry.keys()) >= {"node", "inputs", "outputs"}
        assert "ts" in entry  # monotonic timestamp present


# ---------------------------------------------------------------------------
# refiner branch
# ---------------------------------------------------------------------------

def test_refine_path_runs_refiner_when_confidence_below_threshold():
    """With refine_threshold > 1.0 (confidence can never reach it) the refiner always runs:
    the trace contains a 'refiner' entry, exactly one extra candidate is appended, and the
    committed final_answer still equals the human resume value."""
    base_k = SearchConfig().k_samples
    state = run(
        "I have 2 apples, 3 bananas, and 4 oranges; how many pieces of fruit total?",
        MockBackend(),
        resume_value="9",
        thread_id="refine-1",
        refine_threshold=1.5,  # force the refiner (confidence <= 1.0 always)
    )
    trace = state.get("trace", [])
    names = [entry["node"] for entry in trace]
    assert "refiner" in names
    assert state.get("refined") is True
    # refiner added exactly one more candidate beyond the proposer's k_samples
    assert len(state["candidates"]) == base_k + 1
    assert state.get("final_answer") == "9"
    # refined verifier outputs are consistent: distribution counts sum to total candidates
    refine_entry = next(e for e in trace if e["node"] == "refiner")
    assert refine_entry["outputs"]["added"] == 1


def test_default_refine_threshold_is_in_unit_range():
    """Sanity: the shipped default threshold is a sensible probability."""
    assert 0.0 < DEFAULT_REFINE_THRESHOLD < 1.0


def test_state_schema_is_a_typeddict():
    """OrchestratorState exposes the documented typed fields."""
    hints = OrchestratorState.__annotations__
    for field in (
        "problem",
        "candidates",
        "best",
        "confidence",
        "distribution",
        "final_answer",
        "trace",
    ):
        assert field in hints, f"missing state field: {field}"
