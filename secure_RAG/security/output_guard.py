def validate_output(answer: str) -> str:
    forbidden = [
        "system prompt",
        "developer instructions",
        "internal policy"
    ]

    for f in forbidden:
        if f in answer.lower():
            raise ValueError("Sensitive content detected in output")

    if len(answer) > 1000:
        raise ValueError("Output too long")

    return answer