from pydantic import BaseModel, Field
from typing import List

class TailoredResume(BaseModel):
    matched_skills: List[str] = Field(
        description="Skills from resume that match the job description"
    )
    missing_skills: List[str] = Field(
        description="Important JD skills not present in resume"
    )
    tailored_summary: str = Field(
        description="Professional resume summary tailored to the JD"
    )
    
# This schema:
# Prevents hallucination
# Enables ATS integration
# Makes downstream automation safe