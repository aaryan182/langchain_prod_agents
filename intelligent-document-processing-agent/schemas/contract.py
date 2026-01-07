from pydantic import BaseModel
from schemas.base import BaseExtraction

class ContractExtraction(BaseExtraction):
    parties: list[str]
    start_date: str | None
    end_date: str | None
    governing_law: str | None
    