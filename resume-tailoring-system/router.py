# router.py
"""
Cost aware and failure safe routing for resume tailoring.

Responsibilities:
- Decide complexity (simple vs deep)
- Apply retry logic
- Apply fallback logic ONLY where appropriate
"""

from chains.analyzer_chain import build_analyzer_chain
from chains.simple_tailor_chain import build_simple_chain
from chains.deep_tailor_chain import build_deep_chain

from resilience import with_retry, with_fallback


def build_router():
    """
    Builds the routing function used by the application.

    This function owns:
    - Cost decisions
    - Failure behavior
    """

    analyzer = build_analyzer_chain()

    # ---------- Base chains ----------
    simple_chain = build_simple_chain()
    deep_chain = build_deep_chain()

    # Simple chain: retry only (cheap + fast)
    safe_simple_chain = with_retry(simple_chain, max_attempts=2)

    # Deep chain: retry + fallback to simple
    safe_deep_chain = with_fallback(
        primary_chain=with_retry(deep_chain, max_attempts=3),
        fallback_chain=safe_simple_chain
    )

    def route(resume: str, jd: str):
        """
        Routes request based on complexity.

        Guarantees:
        - Structured output
        - Safe degradation
        - No silent failure
        """

        decision = analyzer.invoke({
            "resume": resume,
            "jd": jd
        }).content.strip().lower()

        if decision == "simple":
            return safe_simple_chain.invoke({
                "resume": resume,
                "jd": jd
            })

        # default → deep
        return safe_deep_chain.invoke({
            "resume": resume,
            "jd": jd
        })

    return route
