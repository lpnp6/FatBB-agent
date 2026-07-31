CREATE EXTENSION IF NOT EXISTS pg_search;

CREATE TABLE IF NOT EXISTS "{{table_name}}" (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    source JSONB NOT NULL DEFAULT '{}'::jsonb,
    start_offset INTEGER,
    end_offset INTEGER,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS "{{table_name}}_document_id_idx"
    ON "{{table_name}}" (document_id);

CREATE INDEX IF NOT EXISTS "{{table_name}}_metadata_idx"
    ON "{{table_name}}" USING GIN (metadata jsonb_path_ops);

CREATE INDEX IF NOT EXISTS "{{table_name}}_bm25_idx"
    ON "{{table_name}}"
    USING bm25 (id, content)
    WITH (key_field = 'id');
