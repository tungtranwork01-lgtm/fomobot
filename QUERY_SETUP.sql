-- Chạy script này trong Supabase SQL Editor (Dashboard -> SQL Editor)
-- Tạo 2 hàm RPC để bot có thể:
--   1. Đọc schema thật của CSDL (bảng, cột, kiểu dữ liệu)
--   2. Chạy câu SQL SELECT an toàn (chỉ đọc, không sửa/xóa)

-- ============================================================
-- 1. Hàm lấy schema: trả về tất cả bảng + cột trong schema public
-- ============================================================
CREATE OR REPLACE FUNCTION get_schema_info()
RETURNS TABLE(
  table_name text,
  column_name text,
  data_type text,
  is_nullable text
)
LANGUAGE sql
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT
    c.table_name::text,
    c.column_name::text,
    c.data_type::text,
    c.is_nullable::text
  FROM information_schema.columns c
  JOIN information_schema.tables t
    ON c.table_name = t.table_name
    AND c.table_schema = t.table_schema
  WHERE c.table_schema = 'public'
    AND t.table_type = 'BASE TABLE'
  ORDER BY c.table_name, c.ordinal_position;
$$;

-- ============================================================
-- 2. Hàm chạy SQL read-only: chỉ cho phép SELECT/WITH, chặn mọi thao tác ghi
-- ============================================================
CREATE OR REPLACE FUNCTION execute_readonly_sql(query text)
RETURNS json
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  result json;
  normalized text;
BEGIN
  normalized := lower(btrim(query));

  -- Phải bắt đầu bằng SELECT hoặc WITH (CTE)
  IF NOT (normalized LIKE 'select %' OR normalized LIKE 'with %') THEN
    RAISE EXCEPTION 'Chỉ cho phép câu lệnh SELECT.';
  END IF;

  -- Chặn các từ khoá ghi/xoá/DDL (word boundary \y)
  IF normalized ~ '\y(insert|update|delete|drop|alter|create|truncate|grant|revoke|copy)\y' THEN
    RAISE EXCEPTION 'Câu lệnh chứa từ khoá bị cấm.';
  END IF;

  -- Chạy và trả về JSON array
  EXECUTE format(
    'SELECT coalesce(json_agg(row_to_json(t)), ''[]''::json) FROM (%s) t',
    query
  ) INTO result;

  RETURN result;
END;
$$;

-- ============================================================
-- 3. Bảng RAG: lưu chunk text + embedding (vector) để tìm kiếm ngữ nghĩa
-- ============================================================

-- Bật extension vector (embedding) và tìm kiếm gần đúng (fallback)
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS rag_chunks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  source TEXT,
  content TEXT NOT NULL,
  embedding vector(1536),
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Thêm cột embedding nếu bảng đã tồn tại từ trước (chạy migration)
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'rag_chunks' AND column_name = 'embedding'
  ) THEN
    ALTER TABLE rag_chunks ADD COLUMN embedding vector(1536);
  END IF;
END $$;

-- Index cho tìm kiếm vector (cosine distance)
CREATE INDEX IF NOT EXISTS rag_chunks_embedding_idx
  ON rag_chunks USING hnsw (embedding vector_cosine_ops)
  WHERE embedding IS NOT NULL;

CREATE INDEX IF NOT EXISTS rag_chunks_content_trgm_idx
  ON rag_chunks USING gin (content gin_trgm_ops);

-- ============================================================
-- 4. Hàm tìm chunk theo embedding (vector similarity)
-- query_embedding_text: dạng '[0.1, -0.2, ...]' (1536 số) từ client
-- ============================================================
CREATE OR REPLACE FUNCTION search_rag_by_embedding(
  query_embedding_text text,
  match_count int DEFAULT 10
)
RETURNS TABLE(id uuid, source text, content text, similarity float)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  q vector(1536);
BEGIN
  q := query_embedding_text::vector(1536);
  RETURN QUERY
  SELECT
    c.id, c.source, c.content,
    (1 - (c.embedding <=> q))::float AS similarity
  FROM rag_chunks c
  WHERE c.embedding IS NOT NULL
  ORDER BY c.embedding <=> q
  LIMIT match_count;
END;
$$;

-- Xóa toàn bộ chunk (gọi trước khi re-index)
CREATE OR REPLACE FUNCTION truncate_rag_chunks()
RETURNS void
LANGUAGE sql
SECURITY DEFINER
SET search_path = public
AS $$
  TRUNCATE TABLE rag_chunks;
$$;

-- ============================================================
-- 5. Hàm tìm chunk theo từ khóa (fallback khi chưa có embedding)
-- ============================================================
CREATE OR REPLACE FUNCTION search_rag_chunks(
  keywords text[],
  match_count int DEFAULT 10
)
RETURNS TABLE(id uuid, source text, content text, relevance bigint)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  RETURN QUERY
  SELECT
    c.id, c.source, c.content,
    (SELECT count(*) FROM unnest(keywords) k WHERE c.content ILIKE '%' || k || '%') AS relevance
  FROM rag_chunks c
  WHERE EXISTS (
    SELECT 1 FROM unnest(keywords) k WHERE c.content ILIKE '%' || k || '%'
  )
  ORDER BY relevance DESC, c.created_at DESC
  LIMIT match_count;
END;
$$;
