from schemas.base import BaseExtraction

class InvoiceExtraction(BaseExtraction):
    invoice_number: str | None
    vendor_name: str | None
    total_amount: float | None
    currency: str | None
    due_date: str | None
    
