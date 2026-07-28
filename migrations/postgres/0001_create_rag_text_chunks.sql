CREATE EXTENSION IF NOT EXISTS pg_search;

CREATE TABLE IF NOT EXISTS rag_text_chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    source JSONB NOT NULL DEFAULT '{}'::jsonb,
    start_offset INTEGER,
    end_offset INTEGER,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS rag_text_chunks_document_id_idx
    ON rag_text_chunks (document_id);

CREATE INDEX IF NOT EXISTS rag_text_chunks_metadata_idx
    ON rag_text_chunks USING GIN (metadata jsonb_path_ops);

CREATE INDEX IF NOT EXISTS rag_text_chunks_bm25_idx
    ON rag_text_chunks
    USING bm25 (id, content)
    WITH (key_field = 'id');
