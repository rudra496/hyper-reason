"""LangGraph orchestrator (Phase 4). Builds the proposer->verifier->refiner state graph
with checkpointing, human-in-the-loop interrupt, and a JSONL decision trace.

Populated in Phase 4. The engine core (``hyper_reason.engine``) does NOT depend on this
package, so the browser/test paths stay langgraph-free.
"""
