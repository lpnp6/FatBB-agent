CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE "{{table_name}}"
    ADD COLUMN IF NOT EXISTS embedding vector({{embedding_dimension}});
