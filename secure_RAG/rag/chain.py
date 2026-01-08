# rag/chain.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from rag.prompts import SAFE_PROMPT
from rag.schemas import SecureRAGResponse
from rag.retriever import retrieve_docs
from rag.validator import validate_rag_output

from security.doc_sanitizer import sanitize_document
from security.injection_detector import is_suspicious

def build_secure_rag(store):
    prompt = ChatPromptTemplate.from_template(SAFE_PROMPT)

    llm = ChatOpenAI(
        model="gpt-4.1",
        temperature=0.0
    )

    def run(question: str):
        # Retrieve documents
        docs = retrieve_docs(store, question)

        # Sanitize & filter documents
        clean_docs = []
        for d in docs:
            sanitized = sanitize_document(d.page_content)
            if is_suspicious(sanitized):
                continue
            clean_docs.append(sanitized)

        context = "\n".join(clean_docs)

        # No safe context → abstain
        if not context:
            return SecureRAGResponse(
                answer="I don't know",
                grounded=False
            )

        # LLM call
        response = (prompt | llm).invoke({
            "context": context,
            "question": question
        })

        answer = response.content

        # FINAL VALIDATION (fail closed)
        try:
            validate_rag_output(answer, context)
        except ValueError:
            # Fail closed — never expose unsafe output
            return SecureRAGResponse(
                answer="I don't know",
                grounded=False
            )

        # Safe to return
        return SecureRAGResponse(
            answer=answer,
            grounded=True
        )

    return run
