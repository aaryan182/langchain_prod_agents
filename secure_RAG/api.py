from fastapi import FastAPI

from security.input_guard import validate_user_input
from vectorstore.factory import get_store
from rag.chain import build_secure_rag

app = FastAPI()

@app.post("/secure-rag")
def secure_rag(payload: dict):
    question = validate_user_input(payload["question"])
    tenant_id = payload["tenant_id"]

    store = get_store(tenant_id)
    rag = build_secure_rag(store)

    result = rag(question)

    return result.dict()