def retrieve_docs(store, query: str, k: int = 4):
    return store.similarity_search(query, k = k)

# Retrieval itself is not a security logic
