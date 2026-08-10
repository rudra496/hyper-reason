"""Phase 0 contract tests — verify the locked types/config/math are correct and real.

These assert genuine properties, not hardcoded outputs. They gate every downstream phase.
"""

import math
import pytest

from hyper_reason.backends import Sample, ModelBackend, count_tokens
from hyper_reason.backends.base import _TOKENIZER_LIVE
from hyper_reason.engine import (
    SearchConfig,
    shannon,
    shannon_from_counts,
    compute_salience,
    extract_boxed,
    extract_final_answer,
    normalize_step,
)


class TestSampleContract:
    def test_sample_is_frozen_with_required_fields(self):
        s = Sample(text="hello", prompt_tokens=3, completion_tokens=2)
        assert s.total_tokens == 5
        assert s.finish_reason == "stop"
        with pytest.raises(Exception):  # frozen dataclass
            s.text = "mutated"

    def test_sample_defaults(self):
        s = Sample(text="x")
        assert s.prompt_tokens == 0 and s.completion_tokens == 0
        assert s.total_tokens == 0


class TestModelBackendProtocol:
    def test_protocol_is_runtime_checkable(self):
        class Dummy:
            name = "dummy"
            is_live = False

            def sample(self, prompt, k=1, temperature=0.7, max_tokens=256, stop=()):
                return [Sample(text="ok")]

            def count_tokens(self, text):
                return len(text)

        assert isinstance(Dummy(), ModelBackend)


class TestSearchConfig:
    def test_locked_fields_exist(self):
        c = SearchConfig()
        for f in ("num_simulations", "max_depth", "k_samples", "temperature",
                  "c_puct", "entropy_alpha", "judge_model"):
            assert hasattr(c, f)

    def test_disclosure_is_honest_about_entropy(self):
        d = SearchConfig().disclosure()
        # The honest label, verbatim — must never drift to "token/policy entropy".
        assert "sample_diversity_entropy" in d["entropy_source"]
        assert "no logprobs" in d["entropy_source"]
        assert d["temperature"] == 0.7  # disclosed because entropy is a function of T


class TestMathUtils:
    def test_shannon_fair_coin_is_one_bit(self):
        assert math.isclose(shannon([0.5, 0.5]), 1.0, abs_tol=1e-9)

    def test_shannon_certain_is_zero(self):
        assert shannon([1.0]) == 0.0

    def test_shannon_from_counts(self):
        assert math.isclose(shannon_from_counts([1, 1]), 1.0, abs_tol=1e-9)
        assert shannon_from_counts([5, 0, 0]) == 0.0  # ignores zero buckets
        assert shannon_from_counts([]) == 0.0

    def test_extract_boxed(self):
        assert extract_boxed(r"answer is \boxed{29}") == "29"
        assert extract_boxed(r"\boxed{a} then \boxed{42}") == "42"  # last one
        assert extract_boxed("no box here") is None

    def test_extract_final_answer_fallbacks(self):
        assert extract_final_answer(r"\boxed{7}") == "7"
        assert extract_final_answer("the result is 12") == "12"
        assert extract_final_answer("words only") == "__unparsable__"

    def test_normalize_step_buckets_semantic_duplicates(self):
        a = normalize_step(r"Step 1: Compute 3*4 = \boxed{12}.")
        b = normalize_step(r"step 1: compute 3*4 = \boxed{99}.")  # same shape, diff answer
        assert a == b  # answer stripped -> buckets together (diversity over wording)

    def test_compute_salience_shape_and_formula(self):
        attn = [[0.5, 0.5], [0.0, 1.0]]
        ent = [0.0, 1.0]
        sal = compute_salience(attn, ent, gamma=1.0)
        # col0 mass = 0.5+0.0=0.5 -> *(1+1*0)=0.5 ; col1 mass=0.5+1.0=1.5 -> *(1+1*1)=3.0
        assert math.isclose(sal[0], 0.5, abs_tol=1e-9)
        assert math.isclose(sal[1], 3.0, abs_tol=1e-9)


class TestTokenCounting:
    def test_count_tokens_positive_and_documented(self):
        n = count_tokens("the quick brown fox")
        assert n > 0
        # When tiktoken is absent this is a labeled estimate (~1.3 * words); either way > 0.

    def test_tokenizer_liveness_flag_exposed(self):
        # The simulator must report whether counts are real-tokenizer or heuristic.
        assert isinstance(_TOKENIZER_LIVE, bool)
