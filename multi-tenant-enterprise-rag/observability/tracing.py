def trace(event: str, payload: dict):
    print({
        "event": event,
        **payload
    })