"""Honesty gates — static checks that catch relapse into v1.x-style fabrication.

These fail loudly if someone reintroduces: third-party CDNs on the keyless demo page, a false
"policy/token entropy" claim, unlabeled FlashKV "measurements", or silent fake fallbacks.
"""

import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = open(os.path.join(ROOT, "docs", "index.html"), encoding="utf-8").read()
ENGINE_JS = open(os.path.join(ROOT, "docs", "engine.js"), encoding="utf-8").read()


def test_no_third_party_cdns_on_demo_page():
    """The demo handles a user key in-memory, so no third-party JS/CSS may load (supply-chain safety)."""
    forbidden = ["cdnjs.cloudflare.com", "jsdelivr.net", "unpkg.com",
                 "fonts.googleapis.com", "fonts.gstatic.com"]
    for f in forbidden:
        assert f not in INDEX, f"third-party CDN {f} present in docs/index.html"
    # no external <script src> or <link href> resources at all
    assert not re.search(r'<script\s+src="https?://', INDEX)
    assert not re.search(r'<link[^>]+href="https?://', INDEX)


def test_entropy_is_labeled_as_sample_diversity():
    assert "sample_diversity" in INDEX
    assert "no logprobs" in INDEX  # the honest caveat
    # must NOT claim token/policy entropy (would imply we have logprobs we don't)
    assert "policy entropy" not in INDEX.lower()
    assert "token entropy" not in INDEX.lower()


def test_flashkv_is_labeled_projected():
    """KV savings must be labeled projected/simulated (no real GPU in this stack)."""
    low = INDEX.lower()
    assert "projected" in low and "simulator" in low
    assert "no real gpu" in low


def test_no_silent_fake_fallback_claim():
    """The page must advertise that backends raise when offline (the v1.x sin was a silent fake)."""
    assert "raise" in INDEX.lower() or "raises" in INDEX.lower()


def test_engine_js_has_no_network_calls():
    """The browser engine port must not fetch a model (the demo is keyless by design)."""
    for needle in ["fetch(", "XMLHttpRequest", "new WebSocket"]:
        assert needle not in ENGINE_JS, f"{needle} found in engine.js — demo must be keyless"


def test_readme_publishes_real_numbers_not_sota():
    readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    # the real (unflattering) numbers are present
    assert "95.0%" in readme and "85.0%" in readme
    # and an explicit non-SOTA statement
    assert "SOTA" in readme or "not a SOTA" in readme.lower() or "opposite of a SOTA" in readme.lower()
