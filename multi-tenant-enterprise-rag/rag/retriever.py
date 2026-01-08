def retrieve_context(vectorstore, question: str, k: int = 4) -> str:
    docs = vectorstore.similarity_search(question, k = k)
    return "\n".join(d.page_content for d in docs)