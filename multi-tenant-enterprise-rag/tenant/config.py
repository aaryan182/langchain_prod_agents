from dataclasses import dataclass

@dataclass
class TenantConfig: 
    tenant_id: str
    prompt_version: str
    max_tokens_per_day: int
    model_tier: str #cheap | balanced | premium
    
    