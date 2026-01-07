from fastapi import FastAPI
from worker import process_document

app = FastAPI()

@app.post("/process")
def process(payload: dict):
    document = payload["document"]
    decision = process_document(document)
    return {"status": decision}