CREATE TABLE IF NOT EXISTS tracks (
  track_id TEXT PRIMARY KEY,
  title TEXT,
  artist TEXT,
  preview_url TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS track_embeddings (
  track_id TEXT PRIMARY KEY REFERENCES tracks(track_id) ON DELETE CASCADE,
  song_vec vector(768),
  updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS track_embeddings_song_hnsw
  ON track_embeddings USING hnsw (song_vec vector_cosine_ops);