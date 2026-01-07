from schemas.base import BaseExtraction

class InsuranceExtraction(BaseExtraction):
    policy_number: str | None
    claim_amount: float | None
    claimant_name: str | None
    
