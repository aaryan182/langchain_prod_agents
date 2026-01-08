from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

_STORES = {}

def get_store(tenant_id: str): 
    if tenant_id not in _STORES:
        _STORES[tenant_id] = FAISS.from_texts(
            texts= [],
            embedding= OpenAIEmbeddings()
        )
    return _STORES[tenant_id]