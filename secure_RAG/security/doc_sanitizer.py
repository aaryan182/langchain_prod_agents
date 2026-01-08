import re

SUSPICIOUS_PATTERNS = [
    r"ignore.*instructions",
    r"execute.*command",
    r"call.*tool",
    r"system message",
    r"developer message"
]

def sanitize_document(text: str) -> str:
    sanitized = text
    
    for pattern in SUSPICIOUS_PATTERNS:
        sanitized = re.sub(pattern, "[REDACTED]", sanitized, flags= re.IGNORECASE)
        
    # remove excessive instructions
    sanitized = sanitized.replace("###","")
        
    return sanitized.strip()


# Why sanitization is needed: attackers hide instructions inside documents, LLMs will follow them if not removed