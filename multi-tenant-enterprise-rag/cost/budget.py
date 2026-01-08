_USAGE ={}

def check_and_consume(tenant_id: str, tokens: int, limit: int):
    used = _USAGE.get(tenant_id, 0)
    
    if used + tokens > limit: 
        raise RuntimeError("Tenant token budget exceeded")
    
    _USAGE[tenant_id] = used + tokens