from tenant.registry import get_tenant_config

def resolve_tenant(request: dict): 
    tenant_id = request.get("tenant_id")
    if not tenant_id:
        raise ValueError("Missing tenant_id")
    
    return get_tenant_config(tenant_id)