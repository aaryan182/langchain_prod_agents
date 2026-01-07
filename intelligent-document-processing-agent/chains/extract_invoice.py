import yaml
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from schemas.invoice import InvoiceExtraction


def build_invoice_extractor():
    parser = PydanticOutputParser(
        pydantic_object=InvoiceExtraction
    )

    with open("prompts/invoice.yaml") as f:
        data = yaml.safe_load(f)

    prompt = ChatPromptTemplate.from_template(
        data["template"]
    ).partial(
        format_instructions=parser.get_format_instructions()
    )

    llm = ChatOpenAI(
        model="gpt-4.1",
        temperature=0.0
    )

    return prompt | llm | parser