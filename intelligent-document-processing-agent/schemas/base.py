from pydantic import BaseModel, Field

class BaseExtraction(BaseModel):
    confidence: float = Field(
        description="Confidence score between 0.0 and 1.0"
    )
    
# why => Every document must carry confidence, confidence drivers routing, no confidence no automation