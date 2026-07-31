CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE rag_text_chunks
    ADD COLUMN IF NOT EXISTS embedding vector;
