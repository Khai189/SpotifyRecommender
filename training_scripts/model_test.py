import numpy as np
import pandas as pd
import torch
import random
from pathlib import Path
import sys
from typing import Tuple, Dict

# Add the parent directory to the system path to allow imports from other folders
sys.path.append(str(Path(__file__).resolve().parent.parent))

from training_scripts.training_scripts.ranking import rank_by_distance

def create_temporal_embedding(log_mel: torch.Tensor, num_chunks: int = 10) -> torch.Tensor:
    """
    Creates a more discriminate embedding by splitting the spectrogram into
    temporal chunks and extracting statistics from each.
    """
    if log_mel.ndim != 2:
        raise ValueError("Expected a 2D log-mel spectrogram [mels, time]")

    chunks = torch.chunk(log_mel, chunks=num_chunks, dim=1)
    all_chunk_stats = []
    for chunk in chunks:
        if chunk.numel() == 0:
            continue
        mean = chunk.mean()
        std = chunk.std().clamp_min(1e-6)
        max_val = chunk.max()
        all_chunk_stats.extend([mean.unsqueeze(0), std.unsqueeze(0), max_val.unsqueeze(0)])
    embedding = torch.cat(all_chunk_stats, dim=0)
    return torch.nn.functional.normalize(embedding, dim=0)

def load_all_data_from_subdirs(base_dir: Path) -> Tuple[Dict[str, torch.Tensor], pd.DataFrame]:
    """
    Scans all subdirectories in the base_dir, loads all 'resolved.csv' files
    and all '.npz' feature files into a unified library.
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

    # Concatenate all metadata into a single DataFrame
    metadata_df = pd.concat(all_metadata_dfs, ignore_index=True)
    metadata_df.dropna(subset=['spotify_track_id'], inplace=True)
    metadata_df.drop_duplicates(subset=['spotify_track_id'], keep='first', inplace=True)
    metadata_df = metadata_df.set_index("spotify_track_id")
    print(f"\nLoaded a total of {len(metadata_df)} unique tracks from metadata.")

    # Load features and generate embeddings
    embedding_library = {}
    print(f"Generating embeddings for {len(all_feature_files)} feature files...")
    for feature_file in all_feature_files:
        track_id = feature_file.stem
        if track_id in metadata_df.index:
            try:
                data = np.load(feature_file)
                log_mel = torch.from_numpy(data['log_mel']).float()
                embedding = create_temporal_embedding(log_mel, num_chunks=10)
                embedding_library[track_id] = embedding
            except Exception as e:
                print(f"Could not load or process feature file {feature_file.name}: {e}")
    
    print(f"Successfully created embeddings for {len(embedding_library)} tracks.")
    return embedding_library, metadata_df

def main():
    base_path = Path(__file__).resolve().parent.parent
    resolved_data_dir = base_path / "TrainingData" / "data" / "spotify_resolved"

    # Load all data from all genre subdirectories
    embedding_library, metadata_df = load_all_data_from_subdirs(resolved_data_dir)

    if not embedding_library:
        print("No embeddings were created. Exiting.")
        return

    # --- The Core Recommendation Test ---
    query_track_id = random.choice(list(embedding_library.keys()))
    query_vector = embedding_library[query_track_id]

    query_song_info = metadata_df.loc[query_track_id]
    query_artist = query_song_info['artist_name']
    query_track = query_song_info['track_name']

    print("\n" + "="*50)
    print(f"Query Song: {query_artist} - {query_track}")
    print("="*50 + "\n")

    ranking_library = {tid: vec for tid, vec in embedding_library.items() if tid != query_track_id}
    
    top_k = 20
    recommendations = rank_by_distance(query_vector, ranking_library, top_k=top_k)

    print(f"Top {top_k} recommendations (ranked by Euclidean Distance - lower is better):")
    for i, (track_id, distance) in enumerate(recommendations):
        try:
            song_info = metadata_df.loc[track_id]
            if isinstance(song_info, pd.DataFrame):
                song_info = song_info.iloc[0]
            
            artist = song_info['artist_name']
            track = song_info['track_name']
            print(f"{i+1:2d}. Distance: {distance:.4f} | {artist} - {track}")
        except KeyError:
            print(f"{i+1:2d}. Distance: {distance:.4f} | Unknown Track ID: {track_id}")

if __name__ == "__main__":
    main()
