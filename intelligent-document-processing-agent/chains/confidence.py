import yaml
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def score_confidence(document: str, extraction: str) -> float:
    with open("prompts/confidence.yaml") as f:
        data = yaml.safe_load(f)

    prompt = ChatPromptTemplate.from_template(data["template"])

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.0
    )

    result = (prompt | llm).invoke({
        "document": document,
        "extraction": extraction
    }).content.strip()

    try:
        return float(result)
    except Exception:
        return 0.0