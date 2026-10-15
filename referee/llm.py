"""
LLM Client Interfaces and Implementations for ReFEree.

Supports OpenAI API (with hyperparameters from paper: temp=0.1, top_p=0.9, max_tokens=4),
open-source endpoints, custom callables, and offline mock clients for testing.
"""

from __future__ import annotations
import os
import re
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Tuple, Union


class BaseLLMClient(ABC):
    """Abstract base class for LLM evaluators in ReFEree."""

    @abstractmethod
    def evaluate_criterion(self, system_prompt: str, user_prompt: str) -> int:
        """
        Evaluates a single (segment, criterion) pair.
        Returns:
            1: If the criterion does NOT exist (i.e., factually consistent).
            0: If the criterion DOES exist (i.e., factual inconsistency detected).
        """
        pass

    @staticmethod
    def parse_score(output_text: str) -> int:
        """
        Parses binary score (0 or 1) from LLM output text.
        Paper specifies: 'max new tokens = 4' and prompt asks for '## SCORE (score only):'.
        """
        cleaned = output_text.strip()
        # Look for leading or exact 0 or 1
        match = re.search(r"\b([01])\b", cleaned)
        if match:
            return int(match.group(1))

        # Heuristic fallback if model outputs text like 'No inconsistency' -> 1
        lower = cleaned.lower()
        if "1" in lower or "consistent" in lower or "pass" in lower or "does not exist" in lower:
            return 1
        if "0" in lower or "inconsistent" in lower or "fail" in lower or "exists" in lower:
            return 0
        return 1


def load_env_file(env_path: Optional[str] = None) -> None:
    """Loads environment variables from a .env file if present in cwd or specified path."""
    target_path = env_path or os.path.join(os.getcwd(), ".env")
    if os.path.isfile(target_path):
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#") and "=" in stripped:
                        key, val = stripped.split("=", 1)
                        k = key.strip()
                        v = val.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass


class OpenAILLMClient(BaseLLMClient):
    """
    OpenAI API client implementing ReFEree's segment-level criterion evaluator.
    Hyperparameters from Section 4.3:
        temperature = 0.1
        top_p = 0.9
        max_tokens = 4
    """

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 4,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required to use OpenAILLMClient. "
                "Install it with `pip install openai`."
            )

        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        load_env_file()
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(
            api_key=resolved_key if resolved_key else None,
            base_url=base_url,
        )

    def evaluate_criterion(self, system_prompt: str, user_prompt: str) -> int:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        content: str = response.choices[0].message.content or ""
        return self.parse_score(content)


class CallableLLMClient(BaseLLMClient):
    """Allows using any user-defined function as an evaluator."""

    def __init__(self, func: Callable[[str, str], Union[int, str]]):
        self.func = func

    def evaluate_criterion(self, system_prompt: str, user_prompt: str) -> int:
        result = self.func(system_prompt, user_prompt)
        if isinstance(result, int):
            return 1 if result == 1 else 0
        return self.parse_score(str(result))


class MockLLMClient(BaseLLMClient):
    """
    Mock LLM client for offline unit testing, reproducible benchmarks,
    and running without OpenAI API credentials.
    """

    def __init__(
        self,
        default_score: int = 1,
        rule_map: Optional[Dict[Tuple[str, str], int]] = None,
    ):
        self.default_score = default_score
        # Map: (segment_substring, criterion_name_or_id) -> score (0 or 1)
        self.rule_map = rule_map or {}

    def set_rule(self, segment_substring: str, criterion_key: str, score: int) -> None:
        self.rule_map[(segment_substring.lower(), criterion_key.lower())] = score

    def evaluate_criterion(self, system_prompt: str, user_prompt: str) -> int:
        # Check rule map
        user_lower = user_prompt.lower()
        sys_lower = system_prompt.lower()

        for (seg_sub, crit_key), score in self.rule_map.items():
            if seg_sub in user_lower and (crit_key in sys_lower or crit_key in user_lower):
                return score

        return self.default_score

