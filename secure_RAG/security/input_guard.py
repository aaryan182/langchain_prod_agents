import re

JAILBREAK_PATTERNS = {
    r"ignore previous instructions",
    r"system prompt",
    r"you are chatgpt",
    r"act as",
    r"developer message"
}

def validate_user_input(question: str):
    lowered = question.lower()
    
    for pattern in JAILBREAK_PATTERNS:
        if re.search(pattern, lowered):
            raise ValueError("Potential prompt injection attempt detected")
    
    if len(question) > 2000:
        raise ValueError("Input too long")

    return question


# Why this exists: stops obvious jailbreaks, cheap and fast, reduces attack surface early