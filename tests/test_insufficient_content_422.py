import fitz
from fastapi.testclient import TestClient

from pdf_summarizer.api.main import app


def _blank_pdf_bytes() -> bytes:
    doc = fitz.open()
    doc.new_page()
    data = doc.write()
    doc.close()
    return data


def test_sync_summarize_pdf_short_content_returns_422() -> None:
    client = TestClient(app)

    pdf_bytes = _blank_pdf_bytes()
    files = {"file": ("blank.pdf", pdf_bytes, "application/pdf")}
    data = {"language": "zh", "max_length": "500"}

    r = client.post("/api/v1/summarize", files=files, data=data)
    assert r.status_code == 422

    payload = r.json()
    assert payload["success"] is False
    assert payload["error"] == "ContentTooShort"
    assert "至少需要" in payload["message"]
