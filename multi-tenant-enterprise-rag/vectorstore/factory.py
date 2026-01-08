from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

_VECTORSTORES = {}

def get_vectorstore(tenant_id: str):
    if tenant_id not in _VECTORSTORES:
        embeddings = OpenAIEmbeddings()
        _VECTORSTORES[tenant_id] = FAISS.from_texts(
            texts=[],
            embedding = embeddings
        )
        
    return _VECTORSTORES[tenant_id]