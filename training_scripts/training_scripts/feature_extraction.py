import torch
import torchaudio

def logmel_stats_embedding(
    waveform: torch.Tensor,
    sample_rate: int,
    target_sr: int = 22050,
    n_mels: int = 96,
    n_fft: int = 2048,
    hop_length: int = 512,
    f_min: float = 30.0,
) -> torch.Tensor:
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != target_sr:
        waveform = torchaudio.transforms.Resample(sample_rate, target_sr)(waveform)
        sample_rate = target_sr

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=sample_rate / 2.0,
        power=2.0,
    )(waveform)

    x = torch.log(mel.clamp_min(1e-10)).squeeze(0)
    mean = x.mean(dim=-1)
    std = x.std(dim=-1).clamp_min(1e-6)

    emb = torch.cat([mean, std], dim=0)
    return torch.nn.functional.normalize(emb, dim=0)