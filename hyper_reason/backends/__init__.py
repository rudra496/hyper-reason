"""Model backends for HyperReason.

A *backend* is the single place that talks to a language model. The engine never imports
torch/requests/anthropic directly — it calls ``backend.sample(...)``. This makes the engine
honest (the model is real and pluggable) and keeps the browser/test paths dependency-free
(via the deterministic MockBackend).

Availability is stated explicitly per backend via ``is_live``:
  True  -> calls a real model (GLM, Ollama, Transformers)
  False -> deterministic in-process sampler, clearly labeled (Mock)
"""

from .base import Sample, ModelBackend, count_tokens

__all__ = ["Sample", "ModelBackend", "count_tokens"]
