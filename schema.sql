DROP TABLE IF EXISTS embeddings CASCADE;
DROP TABLE IF EXISTS tracks CASCADE;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE tracks (
    id BIGSERIAL PRIMARY KEY,
    spotify_track_id VARCHAR(22) NOT NULL UNIQUE,
    track_name TEXT NOT NULL,
    artist_name TEXT NOT NULL,
    spotify_url TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_tracks_spotify_track_id ON tracks(spotify_track_id);


CREATE TABLE embeddings (
    id BIGSERIAL PRIMARY KEY,
    track_id BIGINT NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    embedding_vector VECTOR(250) NOT NULL, -- Corrected size
    embedding_type VARCHAR(50) NOT NULL, -- e.g., 'temporal_v1', 'temporal_v2_librosa'
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_embeddings_track_id ON embeddings(track_id);


CREATE INDEX idx_embeddings_hnsw ON embeddings USING hnsw (embedding_vector vector_l2_ops);
