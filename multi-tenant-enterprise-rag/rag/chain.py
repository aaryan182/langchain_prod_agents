from langchain_openai import ChatOpenAI
from langchain_core.concepts import ChatPromptTemplate
from rag.schemas import RAGResponse
from rag.prompts import get_prompt
from rag.retriever import retrieve_context
from cost.budget import check_and_consume

def build_rag_chain(tenant_config, vectorstore):
    prompt = ChatPromptTemplate.from_template(
        get_prompt(tenant_config.prompt_version)
    )
    
    model_name = {
        "cheap": "gpt-4.1-mini",
        "balanced": "gpt-4.1",
        "premium": "gpt-4.1"
    }[tenant_config.model_tier]

    llm = ChatOpenAI(
        model=model_name,
        temperature=0.0
    )
    
    def run(question: str):
        context = retrieve_context(vectorstore, question)

        response = (prompt | llm).invoke({
            "context": context,
            "question": question
        })

        # Conservative token estimate
        check_and_consume(
            tenant_config.tenant_id,
            tokens=len(response.content.split()),
            limit=tenant_config.max_tokens_per_day
        )

        return RAGResponse(
            answer=response.content,
            confidence=0.85 if context else 0.3
        )

    return run
    