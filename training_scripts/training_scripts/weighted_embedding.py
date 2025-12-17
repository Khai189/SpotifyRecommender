import torch
import torchaudio
from .feature_extraction import *
from .rhythm_embedding import *

def stem_embed_from_path(wav_path: str, max_seconds: float = 60.0) -> tuple[torch.Tensor, torch.Tensor]:
    waveform, sr = torchaudio.load(wav_path)

    # trim for consistency/speed
    max_len = int(max_seconds * sr)
    waveform = waveform[:, :max_len]

    timbre = logmel_stats_embedding(waveform, sr)
    rhythm = rhythm_embedding(waveform, sr)
    return timbre, rhythm


def song_embedding_from_stems(stems: dict[str, str]) -> torch.Tensor:
    """
    stems: {"drums": "...wav", "bass": "...wav", "other": "...wav"}
    Returns one fixed-size vector.
    """
    parts = []

    for stem_name in ["drums", "bass", "other"]:
        timbre, rhythm = stem_embed_from_path(stems[stem_name])

        if stem_name == "drums":
            vec = torch.cat([0.35 * timbre, 0.65 * rhythm], dim=0)
        elif stem_name == "bass":
            vec = torch.cat([0.85 * timbre, 0.15 * rhythm], dim=0)
        else:  # other
            vec = torch.cat([0.70 * timbre, 0.30 * rhythm], dim=0)

        parts.append(vec)

    song_vec = torch.cat(parts, dim=0)
    return torch.nn.functional.normalize(song_vec, dim=0)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b).item())