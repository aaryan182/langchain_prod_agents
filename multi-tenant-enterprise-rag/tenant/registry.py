from tenant.config import TenantConfig

_TENANTS = {
    "acme": TenantConfig(
        tenant_id="acme",
        prompt_version="v1",
        max_tokens_per_day=50_000,
        model_tier="balanced"
    ),
    "globex": TenantConfig(
        tenant_id="globex",
        prompt_version="v2",
        max_tokens_per_day=200_000,
        model_tier="premium"
    ),
}

def get_tenant_config(tenant_id: str) -> TenantConfig:
    if tenant_id not in _TENANTS:
        raise ValueError("unknown tenant")
    return _TENANTS[tenant_id]