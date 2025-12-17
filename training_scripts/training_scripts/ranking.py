import torch
import torchaudio
from .rhythm_embedding import *
from .feature_extraction import *
from .weighted_embedding import *

def rank_similar(query_vec: torch.Tensor, library: dict[str, torch.Tensor], top_k: int = 10):
    scored = [(song_id, cosine_sim(query_vec, vec)) for song_id, vec in library.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]