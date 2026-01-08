def is_suspicious(text: str) -> bool:
    keywords = [
        "ignore",
        "override",
        "disregard",
        "system",
        "instruction"
    ]
    
    score = sum(1 for k in keywords if k in text.lower())
    
    return score >= 2

# why this exists: regex alone is not enough, defense in depth