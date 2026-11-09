"""Tests for LLMEvaluator provider handling."""

from __future__ import annotations

import pytest

from caliche import LLMEvaluator


def test_unknown_provider_raises_value_error() -> None:
    with pytest.raises(ValueError, match="^Unknown provider: nope. Supported providers: "):
        LLMEvaluator("Check if output is correct", "nope")


def test_evaluate_without_a_provider_refuses() -> None:
    evaluator = LLMEvaluator("Check if output is correct")
    with pytest.raises(
        RuntimeError,
        match="^Evaluation failed: LLMEvaluator needs a provider and does not pick one "
        "from the environment$",
    ):
        evaluator.evaluate("Hello", "Hello")
