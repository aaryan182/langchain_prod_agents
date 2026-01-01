"""
Centralized resilience utilities for LLM chains.

This file exists to ensure:
- Bounded retries for transient failures
- Safe fallbacks for persistent failures
- No silent corruption
- Explicit degradation paths

This pattern is REQUIRED in production LLM systems.
"""

from langchain_core.runnables import RunnableRetry, RunnableLambda
from langchain_openai import ChatOpenAI
from typing import Callable
import logging

logger = logging.getLogger(__name__)


def with_retry(
    chain,
    max_attempts: int = 3
):
    """
    Wraps a Runnable with bounded retry logic.

    Retries are meant ONLY for transient failures:
    - network timeouts
    - rate limits
    - temporary parsing failures

     NOT for logical errors
     NOT infinite
    """

    return RunnableRetry(
        chain,
        max_attempts=max_attempts,
        wait_exponential_jitter=True
    )


def build_fallback_llm():
    """
    Fallback model used when the primary chain fails.

    Design principles:
    - Cheaper than primary
    - Faster
    - Deterministic
    - Still schema-compatible
    """

    return ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.0,
        timeout=20
    )


def with_fallback(
    primary_chain,
    fallback_chain
):
    """
    Wraps two chains into a single safe execution unit.

    Behavior:
    1. Try primary chain (with retry)
    2. If it fails after retries → fallback
    3. If fallback fails → raise hard error

    This prevents:
    - Silent failures
    - Partial outputs
    - Corrupt downstream data
    """

    def safe_execute(input_data):
        try:
            return primary_chain.invoke(input_data)

        except Exception as primary_error:
            logger.warning(
                "Primary chain failed, invoking fallback",
                exc_info=primary_error
            )

            try:
                return fallback_chain.invoke(input_data)

            except Exception as fallback_error:
                logger.error(
                    "Fallback chain also failed",
                    exc_info=fallback_error
                )
                raise RuntimeError(
                    "Both primary and fallback chains failed"
                ) from fallback_error

    return RunnableLambda(safe_execute)