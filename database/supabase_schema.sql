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

-- Enable the vector extension to use the 'vector' type
-- This operation only needs to be executed once per database.
CREATE EXTENSION IF NOT EXISTS vector;

-- Table to store document embeddings
-- Each row represents a fragment of a processed document.
CREATE TABLE documentos_embeddings (
  id BIGSERIAL PRIMARY KEY,
  fuente TEXT, -- Source of the data (e.g. the name of the Excel file)
  fragmento TEXT, -- The text of the fragment/chunk
  embedding VECTOR(768) -- The embedding vector corresponding to the fragment
);

-- Function to search documents by embedding similarity
CREATE OR REPLACE FUNCTION match_documentos (
  query_embedding VECTOR(768),
  match_threshold FLOAT,
  match_count INT
)
RETURNS TABLE (
  id BIGINT,
  fuente TEXT,
  fragmento TEXT,
  similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    documentos_embeddings.id,
    documentos_embeddings.fuente,
    documentos_embeddings.fragmento,
    1 - (documentos_embeddings.embedding <=> query_embedding) AS similarity
  FROM documentos_embeddings
  WHERE 1 - (documentos_embeddings.embedding <=> query_embedding) > match_threshold
  ORDER BY similarity DESC
  LIMIT match_count;
END;
$$;

-- NEW TABLE: Conversation History
-- Stores each exchange to give memory to the agent.
CREATE TABLE conversacion_historial (
    id BIGSERIAL PRIMARY KEY,
    id_usuario TEXT NOT NULL, -- Unique user identifier (e.g. WhatsApp number)
    rol TEXT NOT NULL, -- 'user' or 'ai'
    contenido TEXT, -- The text of the message
    creado_en TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- Create an index to accelerate the search for histories by user
CREATE INDEX IF NOT EXISTS idx_conversacion_historial_id_usuario ON conversacion_historial (id_usuario);
