"""
Evaluation Framework Wrapper

Provides Python-friendly wrappers for evaluation features.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from caliche_runtime import _caliche_py as _runtime
else:
    from caliche._runtime import load_runtime

    _runtime = load_runtime()

PyContainsEvaluator = _runtime.PyContainsEvaluator
PyEvaluationDataset = _runtime.PyEvaluationDataset
PyExactMatchEvaluator = _runtime.PyExactMatchEvaluator
PyLLMEvaluator = _runtime.PyLLMEvaluator


class ExactMatchEvaluator:
    """Exact match evaluator for comparing outputs."""

    def __init__(self, case_sensitive: bool = True):
        """Create an exact match evaluator.

        Args:
            case_sensitive: Whether comparison should be case-sensitive
        """
        self._evaluator = PyExactMatchEvaluator(case_sensitive)

    def evaluate(
        self,
        output: str,
        expected: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate an output against expected output (synchronous wrapper).

        Returns:
            Dictionary with 'score', 'passed', 'details', and optionally 'error' keys.
        """
        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                "Cannot call evaluate from async context. Use evaluate_async() instead."
            )
        except RuntimeError as e:
            if "Cannot call evaluate" in str(e):
                # Re-raise our custom error
                raise
            return asyncio.run(self._evaluate_impl(output, expected, context))

    async def evaluate_async(
        self,
        output: str,
        expected: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate an output against expected output (async version).

        Returns:
            Dictionary with 'score', 'passed', 'details', and optionally 'error' keys.
        """
        return await self._evaluate_impl(output, expected, context)

    async def _evaluate_impl(
        self,
        output: str,
        expected: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Internal implementation of evaluate."""
        loop = asyncio.get_running_loop()
        return await self._evaluator.evaluate(loop, output, expected, context)

    def name(self) -> str:
        """Get evaluator name."""
        return self._evaluator.name()


class ContainsEvaluator:
    """Contains evaluator for checking if output contains expected content."""

    def __init__(self, case_sensitive: bool = False):
        """Create a contains evaluator.

        Args:
            case_sensitive: Whether comparison should be case-sensitive
        """
        self._evaluator = PyContainsEvaluator(case_sensitive)

    def evaluate(
        self,
        output: str,
        expected: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate an output against expected output (synchronous wrapper).

        Returns:
            Dictionary with 'score', 'passed', 'details', and optionally 'error' keys.
        """
        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                "Cannot call evaluate from async context. Use evaluate_async() instead."
            )
        except RuntimeError as e:
            if "Cannot call evaluate" in str(e):
                # Re-raise our custom error
                raise
            return asyncio.run(self._evaluate_impl(output, expected, context))

    async def evaluate_async(
        self,
        output: str,
        expected: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate an output against expected output (async version).

        Returns:
            Dictionary with 'score', 'passed', 'details', and optionally 'error' keys.
        """
        return await self._evaluate_impl(output, expected, context)

    async def _evaluate_impl(
        self,
        output: str,
        expected: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Internal implementation of evaluate."""
        loop = asyncio.get_running_loop()
        return await self._evaluator.evaluate(loop, output, expected, context)

    def name(self) -> str:
        """Get evaluator name."""
        return self._evaluator.name()


class LLMEvaluator:
    """LLM-based evaluator for sophisticated evaluation."""

    def __init__(
        self,
        criteria: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Create an LLM-based evaluator.

        Args:
            criteria: Evaluation criteria description
            provider: LLM provider name. Optional here, but ``evaluate``
                fails without one instead of picking a provider from the
                environment.
            model: Model name (optional)

        Raises:
            ValueError: ``provider`` is not a known provider name.
        """
        self._evaluator = PyLLMEvaluator(criteria, provider, model)

    def evaluate(
        self,
        output: str,
        expected: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate an output against expected output (synchronous wrapper).

        Returns:
            Dictionary with 'score', 'passed', 'details', and optionally 'error' keys.
        """
        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                "Cannot call evaluate from async context. Use evaluate_async() instead."
            )
        except RuntimeError as e:
            if "Cannot call evaluate" in str(e):
                # Re-raise our custom error
                raise
            return asyncio.run(self._evaluate_impl(output, expected, context))

    async def evaluate_async(
        self,
        output: str,
        expected: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate an output against expected output (async version).

        Returns:
            Dictionary with 'score', 'passed', 'details', and optionally 'error' keys.
        """
        return await self._evaluate_impl(output, expected, context)

    async def _evaluate_impl(
        self,
        output: str,
        expected: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Internal implementation of evaluate."""
        loop = asyncio.get_running_loop()
        return await self._evaluator.evaluate(loop, output, expected, context)

    def name(self) -> str:
        """Get evaluator name."""
        return self._evaluator.name()


class EvaluationDataset:
    """Evaluation dataset loader."""

    @staticmethod
    def from_jsonl(path: str) -> Dict[str, Any]:
        """Load dataset from JSONL file (synchronous wrapper).

        Returns:
            Dictionary with 'name', 'description' (optional), and 'test_cases' keys.
        """
        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                "Cannot call from_jsonl from async context. Use from_jsonl_async() instead."
            )
        except RuntimeError as e:
            if "Cannot call from_jsonl" in str(e):
                # Re-raise our custom error
                raise
            return asyncio.run(EvaluationDataset._from_jsonl_impl(path))

    @staticmethod
    async def from_jsonl_async(path: str) -> Dict[str, Any]:
        """Load dataset from JSONL file (async version).

        Returns:
            Dictionary with 'name', 'description' (optional), and 'test_cases' keys.
        """
        return await EvaluationDataset._from_jsonl_impl(path)

    @staticmethod
    async def _from_jsonl_impl(path: str) -> Dict[str, Any]:
        """Internal implementation of from_jsonl."""
        loop = asyncio.get_running_loop()
        return await PyEvaluationDataset.from_jsonl(loop, path)

    @staticmethod
    def from_json(path: str) -> Dict[str, Any]:
        """Load dataset from JSON file (synchronous wrapper).

        Returns:
            Dictionary with 'name', 'description' (optional), and 'test_cases' keys.
        """
        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                "Cannot call from_json from async context. Use from_json_async() instead."
            )
        except RuntimeError as e:
            if "Cannot call from_json" in str(e):
                # Re-raise our custom error
                raise
            return asyncio.run(EvaluationDataset._from_json_impl(path))

    @staticmethod
    async def from_json_async(path: str) -> Dict[str, Any]:
        """Load dataset from JSON file (async version).

        Returns:
            Dictionary with 'name', 'description' (optional), and 'test_cases' keys.
        """
        return await EvaluationDataset._from_json_impl(path)

    @staticmethod
    async def _from_json_impl(path: str) -> Dict[str, Any]:
        """Internal implementation of from_json."""
        loop = asyncio.get_running_loop()
        return await PyEvaluationDataset.from_json(loop, path)
