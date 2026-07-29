from __future__ import annotations

from camel.toolkits.document_processing_toolkit import (
    DocumentProcessingToolkit,
)


class _FakeResponse:
    def __init__(self, *, url, content_type, body=b"", status_code=200):
        self.url = url
        self.headers = {"Content-Type": content_type}
        self.body = body
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size):
        yield self.body[:chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def _toolkit():
    toolkit = object.__new__(DocumentProcessingToolkit)
    toolkit.headers = {"User-Agent": "test"}
    return toolkit


def test_extensionless_pdf_overrides_misleading_html_head(monkeypatch):
    toolkit = _toolkit()
    url = "https://arxiv.org/pdf/1810.04805"

    monkeypatch.setattr(
        "camel.toolkits.document_processing_toolkit.requests.head",
        lambda *args, **kwargs: _FakeResponse(
            url=url,
            content_type="text/html; charset=utf-8",
        ),
    )
    monkeypatch.setattr(
        "camel.toolkits.document_processing_toolkit.requests.get",
        lambda *args, **kwargs: _FakeResponse(
            url=url,
            content_type="application/octet-stream",
            body=b"%PDF-1.7\n",
        ),
    )

    assert toolkit._resolve_document_parser(url) == "pdf"


def test_extensionless_pdf_content_type_does_not_need_get_probe(monkeypatch):
    toolkit = _toolkit()
    url = "https://example.test/download?id=paper"

    monkeypatch.setattr(
        "camel.toolkits.document_processing_toolkit.requests.head",
        lambda *args, **kwargs: _FakeResponse(
            url=url,
            content_type="application/pdf",
        ),
    )

    def unexpected_get(*args, **kwargs):
        raise AssertionError("an unambiguous PDF HEAD response needs no GET")

    monkeypatch.setattr(
        "camel.toolkits.document_processing_toolkit.requests.get",
        unexpected_get,
    )

    assert toolkit._resolve_document_parser(url) == "pdf"


def test_extensionless_html_is_still_a_webpage(monkeypatch):
    toolkit = _toolkit()
    url = "https://example.test/article"

    monkeypatch.setattr(
        "camel.toolkits.document_processing_toolkit.requests.head",
        lambda *args, **kwargs: _FakeResponse(
            url=url,
            content_type="text/html",
        ),
    )
    monkeypatch.setattr(
        "camel.toolkits.document_processing_toolkit.requests.get",
        lambda *args, **kwargs: _FakeResponse(
            url=url,
            content_type="text/html; charset=utf-8",
            body=b"<!doctype html><html><body>article</body></html>",
        ),
    )

    assert toolkit._resolve_document_parser(url) == "webpage"


def test_redirected_final_pdf_suffix_is_respected(monkeypatch):
    toolkit = _toolkit()
    url = "https://example.test/download?id=paper"

    monkeypatch.setattr(
        "camel.toolkits.document_processing_toolkit.requests.head",
        lambda *args, **kwargs: _FakeResponse(
            url="https://cdn.example.test/paper.pdf",
            content_type="application/octet-stream",
        ),
    )

    assert toolkit._resolve_document_parser(url) == "pdf"
