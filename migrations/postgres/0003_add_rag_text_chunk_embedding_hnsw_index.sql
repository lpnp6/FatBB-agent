CREATE INDEX IF NOT EXISTS rag_text_chunks_embedding_hnsw_idx
    ON rag_text_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
