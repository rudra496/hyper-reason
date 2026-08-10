"""LangGraph orchestrator (Phase 4).

Public API:
    build_graph(backend, config=None, checkpointer=None, refine_threshold=...) -> CompiledGraph
    run(problem, backend, config=None, resume_value=None, thread_id="default",
        refine_threshold=...) -> dict
    OrchestratorState: the TypedDict state schema.
    DEFAULT_REFINE_THRESHOLD: confidence below which the refiner node runs.

The graph is proposer -> verifier -> [refiner] -> finalize, compiled with a MemorySaver
checkpointer (resumable) and a human-in-the-loop ``interrupt()`` before the final answer is
committed. Every node appends a dict to an additive ``trace`` channel.

The engine core (``hyper_reason.engine``) does NOT depend on this package, so the engine and
its tests stay langgraph-free. All candidates come from real ``backend.sample()`` calls; answer
selection uses ``self_consistency`` (majority vote over extracted answers, no word-matching
heuristics); entropy provenance is labeled "sample-diversity entropy; no logprobs".
"""

from .graph import (
    DEFAULT_REFINE_THRESHOLD,
    OrchestratorState,
    build_graph,
    run,
)

__all__ = [
    "build_graph",
    "run",
    "OrchestratorState",
    "DEFAULT_REFINE_THRESHOLD",
]
