import stat
import tempfile
import unittest
from pathlib import Path

from fatbb.domain.knowledge_base import KnowledgeBase, KnowledgeBaseConfig
from fatbb.infrastructure.local.local import Local


class LocalTests(unittest.TestCase):
    def test_persists_a_knowledge_base_in_a_private_local_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fatbb" / "knowledge_bases.json"
            local = Local(path)
            knowledge_base = KnowledgeBase(
                id="kb-1",
                name="Docs",
                config=KnowledgeBaseConfig(
                    retrieval_type="bm25", database_type="pg",
                    database_url="postgresql://user:secret@localhost/fatbb", source_type="file_path",
                ),
                source_path="/tmp/docs",
            )

            local.create_knowledge_base(knowledge_base)

            self.assertEqual(local.list_knowledge_bases(), [knowledge_base])
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(path.parent.stat().st_mode), 0o700)

    def test_persists_vector_embedding_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            local = Local(Path(directory) / "knowledge_bases.json")
            knowledge_base = KnowledgeBase(
                id="kb-vector",
                name="Vector Docs",
                config=KnowledgeBaseConfig(
                    retrieval_type="vector",
                    database_type="pg",
                    database_url="postgresql://localhost/fatbb",
                    source_type="file_path",
                    embedding_provider="ollama",
                    embedding_model="nomic-embed-text",
                    embedding_url="http://localhost:11434",
                ),
                source_path="/tmp/docs",
            )

            local.create_knowledge_base(knowledge_base)

            self.assertEqual(local.list_knowledge_bases(), [knowledge_base])
