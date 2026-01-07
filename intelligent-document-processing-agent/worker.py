from chains.classify import build_classifier
from chains.extract_invoice import build_invoice_extractor
from routing.router import route
from storage.results import store_result
from storage.human_queue import enqueue

classifier = build_classifier()
invoice_extractor = build_invoice_extractor()

def process_document(document: str):
    doc_type = classifier.invoke({
        "document": document
    }).content.strip().lower()

    if doc_type == "invoice":
        extraction = invoice_extractor.invoke({
            "document": document
        })

        decision = route(extraction)

        if decision == "auto":
            store_result(extraction)
        else:
            enqueue(document, extraction)

        return decision

    # Unknown or unsupported → human
    enqueue(document, None)
    return "human"