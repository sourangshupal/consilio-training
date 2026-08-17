"""Document loaders for PDF, DOCX, and Markdown files."""

import io


def load_pdf(file_bytes: bytes, filename: str) -> dict:
    import fitz

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = "\n\n".join(page.get_text() for page in doc)
    return {"text": text, "filename": filename, "type": "pdf"}


def load_docx(file_bytes: bytes, filename: str) -> dict:
    from docx import Document

    doc = Document(io.BytesIO(file_bytes))
    text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return {"text": text, "filename": filename, "type": "docx"}


def load_markdown(file_bytes: bytes, filename: str) -> dict:
    text = file_bytes.decode("utf-8")
    return {"text": text, "filename": filename, "type": "markdown"}


LOADERS = {
    "pdf": load_pdf,
    "docx": load_docx,
    "md": load_markdown,
    "markdown": load_markdown,
}


def load_document(uploaded_file) -> dict | None:
    ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
    loader = LOADERS.get(ext)
    if loader is None:
        return None
    return loader(uploaded_file.read(), uploaded_file.name)
