import os
import sys
import psycopg2
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import torch
import numpy as np
import librosa

# Add the training_scripts directory to the system path
# This allows us to import the create_temporal_embedding function
sys.path.append(str(Path(__file__).resolve().parent / "training_scripts"))
from training_scripts.model_test import create_temporal_embedding

# --- Database Connection Details ---
# It's best practice to use environment variables for database credentials
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_NAME = os.environ.get("DB_NAME", "spotify_recommender")
DB_USER = os.environ.get("DB_USER", "husamkm") # Replace with your PostgreSQL username if different
DB_PASSWORD = os.environ.get("DB_PASSWORD", "") # Set this environment variable

def load_all_data_from_subdirs(base_dir: Path) -> Tuple[Dict[str, torch.Tensor], pd.DataFrame]:
    """
    Scans all subdirectories in the base_dir, loads all 'resolved.csv' files
    and all '.npz' feature files, generates embeddings, and returns them.
    This is adapted from model_test.py.
    """
    all_metadata_dfs = []
    all_feature_files = []

    print(f"Scanning for data in subdirectories of: {base_dir}")
    for subdir in base_dir.iterdir():
        if subdir.is_dir():
            metadata_file = subdir / "resolved.csv"
            features_dir = subdir / "features"
            if metadata_file.exists() and features_dir.exists():
                print(f"Found data in '{subdir.name}'")
                all_metadata_dfs.append(pd.read_csv(metadata_file))
                all_feature_files.extend(list(features_dir.glob("*.npz")))

    if not all_metadata_dfs:
        print("No metadata files found.")
        return {}, pd.DataFrame()

    metadata_df = pd.concat(all_metadata_dfs, ignore_index=True)
    metadata_df.dropna(subset=['spotify_track_id'], inplace=True)
    metadata_df.drop_duplicates(subset=['spotify_track_id'], keep='first', inplace=True)
    metadata_df = metadata_df.set_index("spotify_track_id")
    print(f"\nLoaded a total of {len(metadata_df)} unique tracks from metadata.")

    embedding_library = {}
    print(f"Generating embeddings for {len(all_feature_files)} feature files...")
    for feature_file in all_feature_files:
        track_id = feature_file.stem
        if track_id in metadata_df.index:
            try:
                data = np.load(feature_file)
                log_mel = torch.from_numpy(data['log_mel']).float()
                sr = data['sr'].item() if 'sr' in data else 22050
                embedding = create_temporal_embedding(log_mel, sr=sr, num_chunks=10)
                if embedding.numel() > 1:
                    embedding_library[track_id] = embedding
            except Exception as e:
                print(f"Could not load or process feature file {feature_file.name}: {e}")
    
    print(f"Successfully created embeddings for {len(embedding_library)} tracks.")
    return embedding_library, metadata_df.reset_index() # Reset index to iterate over rows

def main():
    """
    Main function to load data, generate embeddings, and populate the database.
    """
    # --- 1. Load Data and Generate Embeddings ---
    base_path = Path(__file__).resolve().parent
    resolved_data_dir = base_path / "TrainingData" / "data" / "spotify_resolved"
    
    embedding_library, metadata_df = load_all_data_from_subdirs(resolved_data_dir)

    if not embedding_library:
        print("No embeddings were created. Exiting.")
        return

    # --- 2. Connect to PostgreSQL Database ---
    conn = None
    try:
        print(f"\nConnecting to database '{DB_NAME}' on host '{DB_HOST}'...")
        conn = psycopg2.connect(
            host=DB_HOST,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cur = conn.cursor()
        print("Connection successful.")

        # --- 3. Populate Database ---
        embedding_type = "librosa_v1"
        print(f"Starting database population with embedding type: '{embedding_type}'")
        
        inserted_count = 0
        for _, row in metadata_df.iterrows():
            spotify_id = row['spotify_track_id']
            
            if spotify_id not in embedding_library:
                continue

            try:
                # Step 3a: Insert track metadata and get the new primary key (id)
                # ON CONFLICT DO NOTHING ensures we don't insert duplicate tracks.
                cur.execute(
                    """
                    INSERT INTO tracks (spotify_track_id, track_name, artist_name, spotify_url)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (spotify_track_id) DO NOTHING
                    RETURNING id;
                    """,
                    (spotify_id, row['track_name'], row['artist_name'], row.get('spotify_url'))
                )
                
                # Fetch the returned id. It might be None if the track already existed.
                track_pk_result = cur.fetchone()
                
                if track_pk_result is None:
                    # If the track already existed, we need to fetch its id
                    cur.execute("SELECT id FROM tracks WHERE spotify_track_id = %s;", (spotify_id,))
                    track_pk_result = cur.fetchone()

                if track_pk_result is None:
                    print(f"Warning: Could not find or insert track for Spotify ID: {spotify_id}")
                    continue
                
                track_pk = track_pk_result[0]

                # Step 3b: Insert the embedding vector
                embedding_vector = embedding_library[spotify_id].numpy()
                
                # Check if this specific embedding already exists for this track
                cur.execute(
                    "SELECT 1 FROM embeddings WHERE track_id = %s AND embedding_type = %s;",
                    (track_pk, embedding_type)
                )
                if cur.fetchone():
                    # This specific embedding version already exists for this track, skip it.
                    continue

                cur.execute(
                    """
                    INSERT INTO embeddings (track_id, embedding_vector, embedding_type)
                    VALUES (%s, %s, %s);
                    """,
                    (track_pk, embedding_vector.tolist(), embedding_type)
                )
                inserted_count += 1

            except Exception as e:
                print(f"Error inserting data for Spotify ID {spotify_id}: {e}")
                conn.rollback() # Rollback the transaction on error

        # Commit all successful transactions
        conn.commit()
        print(f"\nDatabase population complete. Inserted {inserted_count} new embeddings.")

    except psycopg2.Error as e:
        print(f"Database connection error: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    main()
