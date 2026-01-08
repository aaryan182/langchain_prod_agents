from pydantic import BaseModel

class SecureRAGResponse(BaseModel):
    answer: str
    grounded: bool