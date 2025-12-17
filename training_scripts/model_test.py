import numpy as np
import pandas as pd
import torch
import random
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from training_scripts.training_scripts.ranking import rank_similar
from training_scripts.training_scripts.feature_extraction import logmel_stats_embedding

def load_features_and_metadata(features_dir: Path, metadata_csv: Path) -> tuple[dict[str, torch.Tensor], pd.DataFrame]:
    """Loads all .npz features and the metadata CSV."""
    
    # 1. Load metadata
    try:
        metadata_df = pd.read_csv(metadata_csv)
        # Set the track ID as the index for easy lookups
        metadata_df = metadata_df.set_index("spotify_track_id")
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {metadata_csv}")
        return {}, pd.DataFrame()

    # 2. Load all feature files into a dictionary
    feature_library = {}
    print(f"Loading features from: {features_dir}")
    for feature_file in features_dir.glob("*.npz"):
        track_id = feature_file.stem
        if track_id in metadata_df.index:
            try:
                # We use the raw log-mel spectrogram as our feature for now
                data = np.load(feature_file)
                log_mel = torch.from_numpy(data['log_mel']).float()
                
                # For this test, we'll create a simplified embedding.
                # We'll calculate the mean and std across the time axis.
                mean = log_mel.mean(dim=-1)
                std = log_mel.std(dim=-1).clamp_min(1e-6)
                embedding = torch.cat([mean, std], dim=0)
                embedding = torch.nn.functional.normalize(embedding, dim=0)

                feature_library[track_id] = embedding
            except Exception as e:
                print(f"Could not load or process feature file {feature_file.name}: {e}")
        else:
            print(f"Warning: Found feature file for track ID {track_id} but no matching metadata.")
            
    print(f"Successfully loaded {len(feature_library)} tracks into the library.")
    return feature_library, metadata_df

def main():
    # Define paths relative to the project root
    base_path = Path(__file__).resolve().parent.parent
    features_dir = base_path / "TrainingData" / "data" / "spotify_resolved" / "pop" / "features"
    metadata_csv = base_path / "TrainingData" / "data" / "spotify_resolved" / "pop" / "resolved.csv"

    # Load everything into memory
    feature_library, metadata_df = load_features_and_metadata(features_dir, metadata_csv)

    if not feature_library:
        print("No features were loaded. Exiting.")
        return

    # --- The Core Recommendation Test ---

    # 3. Choose a random query song from our library
    query_track_id = random.choice(list(feature_library.keys()))
    query_vector = feature_library[query_track_id]

    # Get the song's name for display
    query_song_info = metadata_df.loc[query_track_id]
    query_artist = query_song_info['artist_name']
    query_track = query_song_info['track_name']

    print("\n" + "="*50)
    print(f"Query Song: {query_artist} - {query_track}")
    print("="*50 + "\n")

    # 4. Rank all songs in the library against the query vector
    # We exclude the query song itself from the library for the ranking
    ranking_library = {tid: vec for tid, vec in feature_library.items() if tid != query_track_id}
    
    top_k = 10
    recommendations = rank_similar(query_vector, ranking_library, top_k=top_k)

    # 5. Display the results in a human-readable format
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
