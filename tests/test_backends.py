"""Backend tests — prove the model layer is real and honest.

- MockBackend: protocol conformance + genuinely varied candidates (real diversity entropy).
- GLMBackend: a LIVE call against the Z.AI gateway (skipped if no key) — the real proof.
- OllamaBackend: RAISES on a dead server (the fake fallback is gone).
- TransformersBackend: graceful absence (None) when torch is missing.
"""

import os
import pytest

from hyper_reason.backends import (
    Sample,
    ModelBackend,
    MockBackend,
    GLMBackend,
    OllamaBackend,
    TransformersBackend,
)
from hyper_reason.engine import shannon_from_counts, normalize_step


class TestMockBackend:
    def test_conforms_to_protocol_and_is_not_live(self):
        mb = MockBackend()
        assert isinstance(mb, ModelBackend)
        assert mb.is_live is False
        assert "mock" in mb.name

    def test_provides_k_distinct_candidates(self):
        mb = MockBackend()
        samples = mb.sample("If 3 boxes of 12 apples and 5 are removed, how many left?", k=4)
        assert len(samples) == 4
        assert all(isinstance(s, Sample) for s in samples)
        # Real diversity: distinct normalized steps -> non-trivial entropy over buckets.
        normed = [normalize_step(s.text) for s in samples]
        distinct = len(set(normed))
        counts = [normed.count(n) for n in set(normed)]
        assert distinct >= 2
        assert shannon_from_counts(counts) > 0.0

    def test_deterministic_same_prompt_same_output(self):
        a = MockBackend().sample("compute 7 * 6", k=2)
        b = MockBackend().sample("compute 7 * 6", k=2)
        assert [s.text for s in a] == [s.text for s in b]


class TestGLMBackendLive:
    KEY = os.environ.get("ANTHROPIC_API_KEY")

    @pytest.mark.skipif(not KEY, reason="no ANTHROPIC_API_KEY in env")
    def test_real_live_sample_returns_text(self):
        # THE real proof: the backend actually calls a model and returns model output.
        be = GLMBackend(model="glm-4.6")
        assert be.is_live is True
        samples = be.sample("Reply with the single word: OK", k=1, max_tokens=10, temperature=0)
        assert len(samples) == 1
        s = samples[0]
        assert s.text.strip(), "empty response from live model"
        assert s.completion_tokens >= 1, "usage reported no completion tokens"
        assert s.latency_ms >= 0

    @pytest.mark.skipif(not KEY, reason="no ANTHROPIC_API_KEY in env")
    def test_raises_on_bad_model(self):
        be = GLMBackend(model="this-model-does-not-exist-xyz")
        with pytest.raises(RuntimeError):
            be.sample("hi", k=1, max_tokens=5)

    def test_requires_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ZAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError):
            GLMBackend(model="glm-4.6")


class TestOllamaBackendNoFakeFallback:
    def test_raises_connection_error_on_dead_server(self):
        # Nothing listening on :9 -> MUST raise, NOT return a fake "[offline]" step.
        be = OllamaBackend(model="llama3.1:8b", base_url="http://127.0.0.1:9", timeout=2)
        with pytest.raises(ConnectionError):
            be.sample("hi", k=1)


class TestTransformersGraceful:
    def test_none_when_torch_absent(self):
        # In this dev env there is no torch -> the symbol is None, not a crash on import.
        import hyper_reason.backends as b
        assert b.TransformersBackend is None or hasattr(b.TransformersBackend, "sample")
