from fastapi import FastAPI
from tenant.middleware import resolve_tenant
from vectorstore.factory import get_vectorstore
from rag.chain import build_rag_chain

app = FastAPI()

@app.post("/rag")
def rag_endpoint(payload: dict):
    tenant = resolve_tenant(payload)
    vectorstore = get_vectorstore(tenant.tenant_id)

    rag = build_rag_chain(tenant, vectorstore)

    result = rag(payload["question"])

    return result.dict()