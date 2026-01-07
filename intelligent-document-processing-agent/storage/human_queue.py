_HUMAN_QUEUE = []

def enqueue(document, extraction):
    _HUMAN_QUEUE.append({
        "document": document,
        "extraction": extraction
    })

def pending():
    return _HUMAN_QUEUE