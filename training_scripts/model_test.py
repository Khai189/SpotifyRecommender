import numpy as np
import pandas as pd
import torch
import random
from pathlib import Path
import sys

# Add the parent directory to the system path to allow imports from other folders
sys.path.append(str(Path(__file__).resolve().parent.parent))

from training_scripts.training_scripts.ranking import rank_similar

def create_spectrogram_embedding(log_mel: torch.Tensor) -> torch.Tensor:
    """
    Creates a more discriminate embedding from a log-mel spectrogram by
    extracting a richer set of statistics from different frequency bands.
    """
    if log_mel.ndim != 2:
        raise ValueError("Expected a 2D log-mel spectrogram [mels, time]")

    # Split the mel bands into three sections: low, mid, high
    num_mels = log_mel.shape[0]
    low_split = num_mels // 4
    mid_split = num_mels * 3 // 4

    bands = {
        "low": log_mel[:low_split, :],
        "mid": log_mel[low_split:mid_split, :],
        "high": log_mel[mid_split:, :]
    }

    all_stats = []
    for band_name, band_data in bands.items():
        if band_data.numel() == 0:
            continue

        # Calculate a richer set of statistics for each band
        mean = band_data.mean()
        std = band_data.std().clamp_min(1e-6)
        max_val = band_data.max()
        
        # Skewness: a measure of asymmetry. Requires careful handling for stability.
        # Formula: E[((X - mu) / sigma)^3]
        skew = ((band_data - mean) / std).pow(3).mean()

        all_stats.extend([mean.unsqueeze(0), std.unsqueeze(0), max_val.unsqueeze(0), skew.unsqueeze(0)])

    # Concatenate all statistics to form the final embedding
    embedding = torch.cat(all_stats, dim=0)
    
    # Normalize the final, richer embedding
    return torch.nn.functional.normalize(embedding, dim=0)

def load_features_and_metadata(features_dir: Path, metadata_csv: Path) -> tuple[dict[str, torch.Tensor], pd.DataFrame]:
    """Loads all .npz features and the metadata CSV, creating rich embeddings."""
    
    try:
        metadata_df = pd.read_csv(metadata_csv)
        metadata_df = metadata_df.set_index("spotify_track_id")
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {metadata_csv}")
        return {}, pd.DataFrame()

    embedding_library = {}
    print(f"Loading features and generating embeddings from: {features_dir}")
    for feature_file in features_dir.glob("*.npz"):
        track_id = feature_file.stem
        if track_id in metadata_df.index:
            try:
                data = np.load(feature_file)
                log_mel = torch.from_numpy(data['log_mel']).float()
                
                embedding = create_spectrogram_embedding(log_mel)
                embedding_library[track_id] = embedding

            except Exception as e:
                print(f"Could not load or process feature file {feature_file.name}: {e}")
        else:
            print(f"Warning: Found feature file for track ID {track_id} but no matching metadata.")
            
    print(f"Successfully created embeddings for {len(embedding_library)} tracks.")
    return embedding_library, metadata_df

def main():
    base_path = Path(__file__).resolve().parent.parent
    features_dir = base_path / "TrainingData" / "data" / "spotify_resolved" / "pop" / "features"
    metadata_csv = base_path / "TrainingData" / "data" / "spotify_resolved" / "pop" / "resolved.csv"

    embedding_library, metadata_df = load_features_and_metadata(features_dir, metadata_csv)

    if not embedding_library:
        print("No embeddings were created. Exiting.")
        return

    query_track_id = random.choice(list(embedding_library.keys()))
    query_vector = embedding_library[query_track_id]

    query_song_info = metadata_df.loc[query_track_id]
    query_artist = query_song_info['artist_name']
    query_track = query_song_info['track_name']

    print("\n" + "="*50)
    print(f"Query Song: {query_artist} - {query_track}")
    print("="*50 + "\n")

    ranking_library = {tid: vec for tid, vec in embedding_library.items() if tid != query_track_id}
    
    top_k = 10
    recommendations = rank_similar(query_vector, ranking_library, top_k=top_k)

    print(f"Top {top_k} recommendations:")
    for i, (track_id, score) in enumerate(recommendations):
        try:
            song_info = metadata_df.loc[track_id]
            artist = song_info['artist_name']
            track = song_info['track_name']
            print(f"{i+1:2d}. Score: {score:.4f} | {artist} - {track}")
        except KeyError:
            print(f"{i+1:2d}. Score: {score:.4f} | Unknown Track ID: {track_id}")

if __name__ == "__main__":
    main()
