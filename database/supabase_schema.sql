-- Vector embeddings table
CREATE TABLE documentos_embeddings (
    id BIGSERIAL PRIMARY KEY,
    fuente TEXT NOT NULL,
    chunk_id TEXT,
    fragmento TEXT NOT NULL,
    embedding VECTOR(768),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create vector similarity search index
CREATE INDEX ON documentos_embeddings USING ivfflat (embedding vector_cosine_ops);

-- Query logs for analytics
CREATE TABLE query_logs (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT,
    query TEXT NOT NULL,
    response TEXT,
    sources JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Function to search for similar document chunks
CREATE OR REPLACE FUNCTION match_documentos (
  query_embedding VECTOR(768),
  match_threshold FLOAT,
  match_count INT
)
RETURNS TABLE (
  id BIGINT,
  fragmento TEXT,
  fuente TEXT,
  similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    de.id,
    de.fragmento,
    de.fuente,
    1 - (de.embedding <=> query_embedding) AS similarity
  FROM
    documentos_embeddings AS de
  WHERE 1 - (de.embedding <=> query_embedding) > match_threshold
  ORDER BY
    similarity DESC
  LIMIT match_count;
END;
$$;
