CREATE TABLE spotify_dataset (
    identifier bigint,
    track_id bit varying [20],
    artist bit varying [20],
    album_name bit varying [30],
    track_name bit varying [30],
    popularity int,
    duration_ms bigint,
    explicit bool,
    danceability float,
    energy float
);
COPY spotify_dataset FROM 'TrainingData/data/dataset.txt' WITH (FORMAT CSV)