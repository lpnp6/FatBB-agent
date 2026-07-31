CREATE INDEX IF NOT EXISTS "{{table_name}}_embedding_hnsw_idx"
    ON "{{table_name}}"
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
