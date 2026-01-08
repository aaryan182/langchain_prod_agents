def audit(event: str, data: dict):
    print({
        "event": event,
        **data
    })