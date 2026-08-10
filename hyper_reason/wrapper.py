"""Universal wrapper — now actually drives the backend.

v1.x ``wrap_model`` stored the model and never called it. This version requires a real
``ModelBackend`` (or ``None`` for the zero-setup deterministic Mock demo) and the returned
``WrappedReasoningModel.reason()`` runs the genuine AE-MCTS search.
"""

from __future__ import annotations

from typing import Any, Optional

from .backends.base import ModelBackend
from .backends.mock_backend import MockBackend
from .engine.config import SearchConfig
from .engine.mcts import ReasonEngine


class WrappedReasoningModel:
    """High-level ``.reason()`` / ``.generate()`` API over a real backend."""

    def __init__(self, backend: ModelBackend, config: Optional[SearchConfig] = None):
        self.backend = backend
        self.config = config or SearchConfig()
        self.engine = ReasonEngine(self.backend, self.config)

    def reason(self, problem: str, num_simulations: Optional[int] = None) -> dict:
        if num_simulations is not None:
            self.engine.config.num_simulations = num_simulations
        return self.engine.reason(problem)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return self.reason(prompt)["solution_trajectory"]


def wrap_model(model: Any = None, config: Optional[SearchConfig] = None) -> WrappedReasoningModel:
    """Wrap a model backend with AE-MCTS test-time search.

    model: a ``ModelBackend`` (e.g. ``GLMBackend()``), or ``None`` for the zero-setup
    deterministic Mock demo (clearly labeled, ``is_live=False``).
    """
    if model is None:
        backend: ModelBackend = MockBackend()
    elif isinstance(model, ModelBackend):
        backend = model
    else:
        raise TypeError(
            "wrap_model(model) expects a ModelBackend (GLMBackend/OllamaBackend/"
            "TransformersBackend/MockBackend) or None for the Mock demo."
        )
    return WrappedReasoningModel(backend=backend, config=config)
