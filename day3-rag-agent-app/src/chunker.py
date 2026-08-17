"""Chunking strategies for document splitting."""

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)


def chunk_documents(
    documents: list[dict],
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    if strategy == "Fixed-size":
        splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=" ",
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )

    chunks = []
    for doc in documents:
        doc_chunks = splitter.split_text(doc["text"])
        chunks.extend(doc_chunks)

    return [c for c in chunks if c.strip()]
