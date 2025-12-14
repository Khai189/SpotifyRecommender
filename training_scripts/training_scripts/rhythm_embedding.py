import torch
import torchaudio

def rhythm_embedding(
    waveform: torch.Tensor,
    sample_rate: int,
    target_sr: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 256,
    max_lag_seconds: float = 4.0,
) -> torch.Tensor:
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != target_sr:
        waveform = torchaudio.transforms.Resample(sample_rate, target_sr)(waveform)
        sample_rate = target_sr

    spec = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
    )(waveform)  # [1, freq, time]

    s = torch.log(spec.clamp_min(1e-10)).squeeze(0)         # [freq, time]
    diff = torch.relu(s[:, 1:] - s[:, :-1])                 # [freq, time-1]
    onset = diff.mean(dim=0)                                # [time-1]
    onset = onset - onset.mean()
    onset = onset / (onset.std().clamp_min(1e-6))

    # Autocorrelation to capture periodicity
    max_lag = int((max_lag_seconds * sample_rate) / hop_length)
    max_lag = min(max_lag, onset.numel() - 1)
    ac = []
    for lag in range(1, max_lag + 1):
        ac.append((onset[:-lag] * onset[lag:]).mean())
    ac = torch.stack(ac)  # [max_lag]

    # Pool autocorr into 64 bins
    bins = 64
    if ac.numel() < bins:
        ac = torch.nn.functional.pad(ac, (0, bins - ac.numel()))
    else:
        ac = torch.nn.functional.interpolate(
            ac.view(1, 1, -1), size=bins, mode="linear", align_corners=False
        ).view(-1)

    return torch.nn.functional.normalize(ac, dim=0)