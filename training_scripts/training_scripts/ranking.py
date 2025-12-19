import torch
import torchaudio
from .rhythm_embedding import *
from .feature_extraction import *
from .weighted_embedding import *

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Calculates cosine similarity between two vectors."""
    return float(torch.dot(a, b).item())

def euclidean_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    """Calculates the straight-line Euclidean distance between two vectors."""
    return float(torch.linalg.norm(a - b).item())

def rank_by_cosine(query_vec: torch.Tensor, library: dict[str, torch.Tensor], top_k: int = 10):
    """Ranks songs by cosine similarity (higher is better)."""
    scored = [(song_id, cosine_sim(query_vec, vec)) for song_id, vec in library.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

def rank_by_distance(query_vec: torch.Tensor, library: dict[str, torch.Tensor], top_k: int = 10):
    """Ranks songs by Euclidean distance (lower is better)."""
    scored = [(song_id, euclidean_dist(query_vec, vec)) for song_id, vec in library.items()]
    scored.sort(key=lambda x: x[1], reverse=False)
    return scored[:top_k]

rank_similar = rank_by_cosine
