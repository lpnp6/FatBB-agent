"""Local Markdown/text source importer."""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
import logging
from pathlib import Path

from rag.models.common import SourceRef
from rag.models.document import Document
from tqdm import tqdm


logger = logging.getLogger(__name__)


class LocalFileImporter:
    type = "file_path"
    _supported_suffixes = {".md", ".markdown", ".txt"}

    def load(
        self, path: str, *,
        knowledge_base_id: str,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> list[Document]:
        source_path = Path(path).expanduser().resolve()
        if not source_path.exists():
            raise ValueError(f"Path does not exist: {source_path}")
        paths = [source_path] if source_path.is_file() else sorted(source_path.rglob("*"))
        supported_paths = [
            item for item in paths
            if item.is_file() and item.suffix.lower() in self._supported_suffixes
        ]
        total = len(supported_paths)
        logger.info("Loading local source files", extra={"file_count": total})
        documents: list[Document] = []
        for idx, item in enumerate(
            tqdm(supported_paths, desc="Loading documents", unit="file", disable=on_progress is not None)
        ):
            if on_progress is not None:
                on_progress("Loading documents", idx + 1, total)
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
        logger.info("Loaded local source documents", extra={"document_count": len(documents)})
        return documents
