CONFIDENCE_THRESHOLD = 0.75

def route(extraction):
    if extraction.confidence >= CONFIDENCE_THRESHOLD:
        return "auto"
    return "human"