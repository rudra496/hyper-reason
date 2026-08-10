"""LangGraph multi-agent orchestrator for hyper_reason (Phase 4).

A real state graph::

    START -> proposer -> verifier -> [refiner] -> finalize -> END

compiled with a ``MemorySaver`` checkpointer (resumable) and a human-in-the-loop
``interrupt()`` inside ``finalize`` so a human can approve/edit the answer before it is
committed. Every node appends one dict to an additive ``trace`` channel
(node name, inputs, outputs, monotonic timestamp).

Honesty contract (non-negotiable):
  * Every candidate string comes from a REAL ``backend.sample()`` call. No fabricated text,
    no hardcoded answers.
  * Answer selection uses ``self_consistency`` from ``hyper_reason.engine.verifier`` (majority
    vote over extracted answers). No word-matching "reward" heuristics.
  * Entropy/diversity provenance is labeled as "sample-diversity entropy (K samples; no
    logprobs via stateless gateway)" wherever it is referenced.
  * No benchmark numbers are baked in.
"""

from __future__ import annotations

import operator
import time
from typing import Annotated, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

from ..backends.base import ModelBackend
from ..engine.config import SearchConfig
from ..engine.mcts import _PROMPT_TEMPLATE
from ..engine.verifier import self_consistency

# If verifier confidence is below this, ask the backend for one more sample and re-verify.
DEFAULT_REFINE_THRESHOLD = 0.75
# Token budget for the single re-check sample the refiner requests.
REFINE_MAX_TOKENS = 256

# Labeled entropy provenance — the package uses sample-diversity entropy; the stateless
# Z.AI gateway exposes no token logprobs, so we never claim logprob-based entropy.
_ENTROPY_NOTE = "sample-diversity entropy (K samples; no logprobs via stateless gateway)"


class OrchestratorState(TypedDict, total=False):
    """Shared state for the orchestrator graph.

    ``candidates`` and ``trace`` use an additive (``operator.add``) reducer so each node
    APPENDS rather than overwrites; all other fields are last-write-wins. ``total=False``
    because the initial input only carries ``problem`` and nodes return partial updates.
    """

    problem: str
    candidates: Annotated[list[str], operator.add]
    best: str
    confidence: float
    distribution: dict[str, int]
    final_answer: str
    refined: bool
    trace: Annotated[list[dict], operator.add]


def _now() -> float:
    """Monotonic timestamp (NOT wall-clock datetime.now) — for trace ordering only."""
    return time.monotonic()


def _proposer_node(backend: ModelBackend, config: SearchConfig):
    """Sample K real candidate continuations of the problem prompt."""

    def node(state: OrchestratorState) -> dict:
        problem = state["problem"]
        prompt = _PROMPT_TEMPLATE.format(problem=problem)
        t0 = _now()
        samples = backend.sample(
            prompt,
            k=config.k_samples,
            temperature=config.temperature,
            max_tokens=config.max_tokens_per_step,
            stop=(),
        )
        texts = [s.text for s in samples]
        elapsed = (_now() - t0) * 1000.0
        return {
            "candidates": texts,
            "trace": [
                {
                    "node": "proposer",
                    "inputs": {"prompt_chars": len(prompt), "k": config.k_samples},
                    "outputs": {
                        "n_candidates": len(texts),
                        "backend": backend.name,
                        "backend_is_live": backend.is_live,
                    },
                    "elapsed_ms": round(elapsed, 3),
                    "ts": _now(),
                }
            ],
        }

    return node


def _verifier_node(backend: ModelBackend):
    """Run self-consistency over the accumulated candidates -> best/confidence/distribution."""

    def node(state: OrchestratorState) -> dict:
        candidates = list(state.get("candidates", []))
        t0 = _now()
        best, confidence, distribution = self_consistency(candidates)
        elapsed = (_now() - t0) * 1000.0
        return {
            "best": best,
            "confidence": confidence,
            "distribution": distribution,
            "trace": [
                {
                    "node": "verifier",
                    "inputs": {"n_candidates": len(candidates)},
                    "outputs": {
                        "best": best,
                        "confidence": confidence,
                        "distribution": distribution,
                        "entropy_source": _ENTROPY_NOTE,
                    },
                    "elapsed_ms": round(elapsed, 3),
                    "ts": _now(),
                }
            ],
        }

    return node


def _refiner_node(backend: ModelBackend, config: SearchConfig):
    """Ask the backend for ONE more sample conditioned on the current best, then re-verify."""

    def node(state: OrchestratorState) -> dict:
        problem = state["problem"]
        best = state.get("best", "")
        confidence = state.get("confidence", 0.0)
        current = list(state.get("candidates", []))
        prompt = (
            "A prior attempt proposed the answer "
            f"{best!r} with self-consistency confidence {confidence}.\n"
            "Carefully re-check the work and put ONLY the final answer in \\boxed{}.\n\n"
            f"Problem: {problem}\n"
        )
        t0 = _now()
        samples = backend.sample(
            prompt,
            k=1,
            temperature=config.temperature,
            max_tokens=REFINE_MAX_TOKENS,
            stop=(),
        )
        extra = [s.text for s in samples]
        # state["candidates"] does not yet include `extra` inside this node, so re-verify on
        # the locally-merged list; the additive reducer appends `extra` to the channel.
        new_candidates = current + extra
        best2, conf2, dist2 = self_consistency(new_candidates)
        elapsed = (_now() - t0) * 1000.0
        return {
            "candidates": extra,  # reducer appends -> accumulated channel becomes current + extra
            "best": best2,
            "confidence": conf2,
            "distribution": dist2,
            "refined": True,
            "trace": [
                {
                    "node": "refiner",
                    "inputs": {"prior_best": best, "prior_confidence": confidence},
                    "outputs": {
                        "added": len(extra),
                        "best": best2,
                        "confidence": conf2,
                        "distribution": dist2,
                        "entropy_source": _ENTROPY_NOTE,
                    },
                    "elapsed_ms": round(elapsed, 3),
                    "ts": _now(),
                }
            ],
        }

    return node


def _finalize_node():
    """Human-in-the-loop gate: interrupt so a human can approve/edit, then commit."""

    def node(state: OrchestratorState) -> dict:
        best = state.get("best", "")
        confidence = state.get("confidence", 0.0)
        # Pause here. On first invoke this raises a GraphInterrupt internally and the node's
        # return is NOT applied. On resume with Command(resume=<value>), the node re-executes
        # and interrupt() returns <value>.
        approved = interrupt({"best": best, "confidence": confidence})
        if isinstance(approved, str):
            final = approved
        elif isinstance(approved, dict):
            # Accept {"answer": ...} or {"best": ...}; fall back to the verified best.
            final = approved.get("answer", approved.get("best", best))
        else:
            final = best
        t0 = _now()
        return {
            "final_answer": final,
            "trace": [
                {
                    "node": "finalize",
                    "inputs": {
                        "best": best,
                        "confidence": confidence,
                        "human_value": approved,
                    },
                    "outputs": {"final_answer": final},
                    "elapsed_ms": round((_now() - t0) * 1000.0, 3),
                    "ts": _now(),
                }
            ],
        }

    return node


def _route_after_verify(threshold: float):
    """Conditional router: low confidence -> refiner, otherwise -> finalize."""

    def router(state: OrchestratorState) -> str:
        return "refiner" if state.get("confidence", 0.0) < threshold else "finalize"

    return router


def build_graph(
    backend: ModelBackend,
    config: Optional[SearchConfig] = None,
    checkpointer=None,
    refine_threshold: float = DEFAULT_REFINE_THRESHOLD,
):
    """Compile the proposer -> verifier -> [refiner] -> finalize state graph.

    Args:
        backend: any ``ModelBackend`` (use ``MockBackend()`` for deterministic tests).
        config: ``SearchConfig`` (default ``SearchConfig()``). ``k_samples``, ``temperature``
            and ``max_tokens_per_step`` drive real sampling.
        checkpointer: a LangGraph checkpointer (default ``MemorySaver()``) for resumability.
            Callers may pass a ``SqliteSaver`` for persistence across processes.
        refine_threshold: if verifier confidence is below this, route through the refiner node
            exactly once before finalizing.

    Returns:
        A compiled langgraph Runnable. Drive it with a thread config::

            cfg = {"configurable": {"thread_id": "t1"}}
            graph.invoke({"problem": "..."}, cfg)               # pauses at finalize
            graph.invoke(Command(resume=<approved>), cfg)        # commits final_answer
    """
    cfg = config or SearchConfig()
    chk = checkpointer if checkpointer is not None else MemorySaver()

    graph = StateGraph(OrchestratorState)
    graph.add_node("proposer", _proposer_node(backend, cfg))
    graph.add_node("verifier", _verifier_node(backend))
    graph.add_node("refiner", _refiner_node(backend, cfg))
    graph.add_node("finalize", _finalize_node())

    graph.add_edge(START, "proposer")
    graph.add_edge("proposer", "verifier")
    graph.add_conditional_edges(
        "verifier",
        _route_after_verify(refine_threshold),
        {"refiner": "refiner", "finalize": "finalize"},
    )
    graph.add_edge("refiner", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=chk)


def run(
    problem: str,
    backend: ModelBackend,
    config: Optional[SearchConfig] = None,
    resume_value=None,
    thread_id: str = "default",
    refine_threshold: float = DEFAULT_REFINE_THRESHOLD,
) -> dict:
    """Build the graph with a ``MemorySaver`` and drive it through the finalize interrupt.

    * Without ``resume_value``: runs proposer -> verifier -> [refiner] -> finalize and returns
      the state at the interrupt. The returned dict has ``awaiting_approval=True`` and NO
      committed ``final_answer``.
    * With ``resume_value``: resumes the finalize interrupt with the (possibly edited)
      human-approved value, committing ``final_answer``.

    Returns the full final state dict (includes ``final_answer``, ``trace``, ``confidence``).
    """
    graph = build_graph(backend, config=config, refine_threshold=refine_threshold)
    thread_config = {"configurable": {"thread_id": thread_id}}

    # Phase 1: always pauses inside finalize (interrupt()).
    state = graph.invoke({"problem": problem}, thread_config)

    if resume_value is None:
        return {**state, "awaiting_approval": True}

    # Phase 2: human-approved value resumes the interrupt and commits final_answer.
    state = graph.invoke(Command(resume=resume_value), thread_config)
    return state
