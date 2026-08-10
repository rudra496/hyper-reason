"""JS<->Python parity test — pins docs/engine.js to the Python package.

Runs the shipped docs/engine.js under Node and asserts its pure functions match the Python
implementations on shared inputs. If this passes, the in-browser demo computes the same
arithmetic as the installed package. Skips if Node is unavailable.
"""

import json
import os
import shutil
import subprocess

import pytest

from hyper_reason.engine import (
    shannon_from_counts, sample_diversity_entropy, priors_from_diversity,
)
from hyper_reason.engine.math_utils import normalize_step, extract_final_answer
from hyper_reason.engine.verifier import self_consistency

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE_JS = os.path.join(ROOT, "docs", "engine.js")
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(
    not (NODE and os.path.exists(ENGINE_JS)),
    reason="node or docs/engine.js not available",
)

CASES = {
    "shannon": [[1, 1], [5, 0, 0], [2, 1, 1], [10], [], [3, 3, 3, 3]],
    "norm": ["Step 1: \\boxed{12}.", "  Hello   World  ", "ABC def", "\\boxed{99} final"],
    "div": [["a", "b", "a"], ["x", "x", "x"], ["\\boxed{1}", "\\boxed{2}"], [], ["a B a"]],
    "prior": [["a", "b", "a"], ["x", "x"], ["p", "q", "r"]],
    "extract": ["\\boxed{42}", "the answer is 7", "no number here",
                "\\boxed{a}\\boxed{b}", "result = -3.5"],
    "sc": [["\\boxed{1}", "\\boxed{1}", "\\boxed{2}"], ["\\boxed{5}"], ["x", "y"]],
}

HARNESS = r"""
const E = require(%r);
const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const out = {
  shannonFromCounts: input.shannon.map(c => E.shannonFromCounts(c)),
  normalizeStep: input.norm.map(t => E.normalizeStep(t)),
  sampleDiversityEntropy: input.div.map(a => E.sampleDiversityEntropy(a)),
  priorsFromDiversity: input.prior.map(a => E.priorsFromDiversity(a)),
  extractFinalAnswer: input.extract.map(t => E.extractFinalAnswer(t)),
  selfConsistency: input.sc.map(a => E.selfConsistency(a)),
};
process.stdout.write(JSON.stringify(out));
""" % ENGINE_JS


def _js_results():
    proc = subprocess.run(
        [NODE, "-e", HARNESS], input=json.dumps(CASES).encode(),
        capture_output=True, timeout=30, cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr.decode()[:500]
    return json.loads(proc.stdout.decode())


def _approx(a, b, tol=1e-9):
    if isinstance(a, float) or isinstance(b, float):
        return abs(a - b) <= tol
    return a == b


def test_shannon_parity():
    js = _js_results()
    for c, jv in zip(CASES["shannon"], js["shannonFromCounts"]):
        assert _approx(shannon_from_counts(c), jv)


def test_normalize_parity():
    js = _js_results()
    for t, jv in zip(CASES["norm"], js["normalizeStep"]):
        assert normalize_step(t) == jv


def test_diversity_entropy_parity():
    js = _js_results()
    for a, jv in zip(CASES["div"], js["sampleDiversityEntropy"]):
        assert _approx(sample_diversity_entropy(a), jv)


def test_priors_parity():
    js = _js_results()
    for a, jv in zip(CASES["prior"], js["priorsFromDiversity"]):
        assert all(_approx(x, y) for x, y in zip(priors_from_diversity(a), jv))


def test_extract_parity():
    js = _js_results()
    for t, jv in zip(CASES["extract"], js["extractFinalAnswer"]):
        assert extract_final_answer(t) == jv


def test_self_consistency_parity():
    js = _js_results()
    for a, jv in zip(CASES["sc"], js["selfConsistency"]):
        ans, conf, dist = self_consistency(a)
        assert ans == jv[0]
        assert _approx(conf, jv[1])
        # distribution: JS object vs Python dict (keys are the answer strings)
        assert dist == {k: v for k, v in jv[2].items()}
