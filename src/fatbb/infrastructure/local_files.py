"""Local Markdown/text source adapter."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from rag.models.common import SourceRef
from rag.models.document import Document


class LocalFileImporter:
    type = "file_path"
    _supported_suffixes = {".md", ".markdown", ".txt"}

    def load(self, path: str, *, knowledge_base_id: str) -> list[Document]:
        source_path = Path(path).expanduser().resolve()
        if not source_path.exists():
            raise ValueError(f"Path does not exist: {source_path}")
        paths = [source_path] if source_path.is_file() else sorted(source_path.rglob("*"))
        documents: list[Document] = []
        for item in paths:
            if not item.is_file() or item.suffix.lower() not in self._supported_suffixes:
                continue
            try:
                content = item.read_text(encoding="utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(f"File is not valid UTF-8: {item}") from error
            if not content.strip():
                continue
            document_id = sha256(f"{knowledge_base_id}:{item}".encode()).hexdigest()
            documents.append(
                Document(
                    id=document_id,
                    content=content,
                    source=SourceRef(document_id=document_id, uri=item.as_uri(), title=item.name),
                    metadata={"knowledge_base_id": knowledge_base_id},
                )
            )
        return documents
