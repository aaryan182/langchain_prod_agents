"""
Final validation layer for Secure RAG outputs.

Responsibilities:
- Enforce groundedness (answer must be supported by context)
- Block leakage of internal/system content
- Enforce length & formatting limits
- Provide a deterministic fail-closed decision
"""

import re

FORBIDDEN_PATTERNS = [
    r"system prompt",
    r"developer instructions",
    r"internal policy",
    r"confidential",
    r"api key",
]

MAX_ANSWER_CHARS = 1200


def _contains_forbidden(answer: str) -> bool:
    lower = answer.lower()
    return any(re.search(p, lower) for p in FORBIDDEN_PATTERNS)


def _is_grounded(answer: str, context: str, min_overlap: int = 2) -> bool:
    """
    Simple lexical grounding check:
    - Count overlapping non trivial tokens between answer and context.
    - This is a conservative heuristic; stronger checks can be added later
      (e.g., entailment models or citation enforcement).
    """
    def tokens(s: str) -> set:
        return {t for t in re.findall(r"[a-zA-Z]{4,}", s.lower())}

    a_tokens = tokens(answer)
    c_tokens = tokens(context)

    return len(a_tokens.intersection(c_tokens)) >= min_overlap


def validate_rag_output(answer: str, context: str) -> None:
    """
    Raises ValueError if validation fails.
    Returns None on success.
    """
    if not answer or not answer.strip():
        raise ValueError("Empty answer")

    if len(answer) > MAX_ANSWER_CHARS:
        raise ValueError("Answer exceeds maximum length")

    if _contains_forbidden(answer):
        raise ValueError("Sensitive or internal content detected")

    # If the model claims knowledge, ensure it's grounded
    if answer.strip().lower() not in {"i don't know", "i do not know"}:
        if not _is_grounded(answer, context):
            raise ValueError("Answer is not sufficiently grounded in context")
