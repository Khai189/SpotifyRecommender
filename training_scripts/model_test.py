import torch
from torch import nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio import datasets
import torchaudio.transforms as transforms
import openunmix
import glob
import os
import torch
import torchaudio
from faster_whisper import WhisperModel




if torch.cuda.is_available():
    device = "cuda"
    print("Cuda device is available")
elif not torch.backends.mps.is_available():
    device= "cpu"
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    device = "mps"
    print("Mac device available")

current_mp3 = 'test.mp3'
audio, rate = torchaudio.load(current_mp3)

if rate != 44100:
    audio = torchaudio.transforms.Resample(rate, 44100)(audio)

audio_tensor = audio.float().unsqueeze(0)
separator = torch.hub.load('sigsep/open-unmix-pytorch', 'umxhq', device="cpu")

with torch.inference_mode():
    estimates = separator(audio_tensor)[0]


stems = ['vocals', 'drums', 'bass', 'other']
output_path = f'{current_mp3}_umx_output'
os.makedirs(output_path, exist_ok=True)

for i, stem in enumerate(stems):
    output_file = os.path.join(output_path, f'umx_{stem}.wav')
    torchaudio.save(output_file, estimates[i].cpu(), 44100)



torch.manual_seed(42)

# Experimentation, we start with an initial torchaudio model and build from there.
# We'll use faster whisper, which is an implementation of ChatGPT's whisper, and hopefully build
# our own custom model if performance sees fit

test_vocal_path = os.path.join(output_path, "umx_vocals.wav")

model_size = "large-v3"

model = WhisperModel(model_size,
                     device="cpu",
                     compute_type="int8"
                     )

segments, info = model.transcribe(test_vocal_path, beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

waveform, sample_rate = torchaudio.load(test_vocal_path)
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
initial_model = bundle.get_model()

print(f"Waveform shape after load: {waveform.shape}")
print(f"Waveform max value: {waveform.max().item()}")
print(f"Waveform min value: {waveform.min().item()}")

initial_model.eval()

if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)
    print(f"Mono shape: {waveform.shape}")
expected_sample_rate = bundle.sample_rate
if sample_rate != expected_sample_rate:
    resampler = torchaudio.transforms.Resample(sample_rate, expected_sample_rate)
    waveform = resampler(waveform)

input_tensor = waveform.squeeze(0)
input_tensor = input_tensor.unsqueeze(0)
with torch.inference_mode():
    logits, _ = initial_model(input_tensor)
labels = bundle.get_labels()
print(f"Labels: {labels}")
BLANK_ID = 0

print(f"Logits shape: {logits.shape}")
print(f"Max logit value: {logits.max().item()}")
print(f"Min logit value: {logits.min().item()}")

# Debugging transcription
emission = logits.squeeze(0)
indices = torch.argmax(emission, dim=-1)
unique_indices = torch.unique_consecutive(indices, dim=-1)
result_indices = [i.item() for i in unique_indices if i != BLANK_ID]
transcript_list = [labels[i] for i in result_indices]
final_text = "".join(transcript_list).replace('|', ' ').strip()
print(f"Raw indices generated: {len(indices)} frames")
print(f"Non-blank/unique characters: {len(result_indices)} characters")
print(f"Transcription: '{final_text}'")

def preprocess_vocals(
    waveform: torch.Tensor,
    sample_rate: int,
    target_sr: int = 16000,
    peak_target: float = 0.98,
    rms_target: float = 0.08,
    apply_vad: bool = True,
    vad_trigger_level: float = 7.0,
) -> tuple[torch.Tensor, int]:

    if waveform.dim() != 2:
        raise ValueError(f"Expected waveform shape [channels, time], got {tuple(waveform.shape)}")

    # Convert to mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample
    if sample_rate != target_sr:
        waveform = torchaudio.transforms.Resample(sample_rate, target_sr)(waveform)
        sample_rate = target_sr

    # Remove DC offset
    waveform = waveform - waveform.mean(dim=-1, keepdim=True)

    peak = waveform.abs().max().clamp_min(1e-8)
    waveform = waveform * (peak_target / peak)

    rms = waveform.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
    waveform = waveform * (rms_target / rms)

    if apply_vad:
        mono_1d = waveform.squeeze(0)
        mono_1d = torchaudio.functional.vad(mono_1d, sample_rate=sample_rate, trigger_level=vad_trigger_level)
        if mono_1d.numel() > 0:
            waveform = mono_1d.unsqueeze(0)

    # Safety clamp
    waveform = waveform.clamp(-1.0, 1.0)
    return waveform, sample_rate

waveform, sample_rate = torchaudio.load(test_vocal_path)
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
initial_model = bundle.get_model()
initial_model.eval()

with torch.inference_mode():
    logits, _ = initial_model(waveform)

labels = bundle.get_labels()
BLANK_ID = 0
print(f"Blank label: {labels[BLANK_ID]!r} at index {BLANK_ID}")

def ctc_beam_search_decode(
    emission: torch.Tensor,
    labels: list[str],
    blank_id: int = 0,
    beam_width: int = 50,
    top_k: int = 1,
) -> list[tuple[str, float]]:

    if emission.dim() != 2:
        raise ValueError(f"Expected emission shape [time, vocab], got {tuple(emission.shape)}")

    # Convert to log-probs robustly
    if emission.dtype != torch.float32 and emission.dtype != torch.float64:
        emission = emission.float()

    # If these aren't log-probs yet, log_softmax makes it safe either way.
    logp = torch.log_softmax(emission, dim=-1)

    def logsumexp(a: float, b: float) -> float:
        # stable log(exp(a) + exp(b))
        if a == -float("inf"):
            return b
        if b == -float("inf"):
            return a
        m = a if a > b else b
        return m + float(torch.log(torch.exp(torch.tensor(a - m)) + torch.exp(torch.tensor(b - m))))

    # beams: prefix -> (p_blank, p_nonblank) in log space
    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, -float("inf"))}

    vocab_size = logp.size(-1)

    for t in range(logp.size(0)):
        next_beams: dict[tuple[int, ...], tuple[float, float]] = {}

        # Take top candidates per timestep to keep it fast-ish
        step_logp = logp[t]
        topv, topi = torch.topk(step_logp, k=min(beam_width, vocab_size))

        for prefix, (p_b, p_nb) in beams.items():
            for k in range(topi.numel()):
                c = int(topi[k].item())
                p = float(topv[k].item())

                if c == blank_id:
                    # stay on same prefix, end with blank
                    nb = next_beams.get(prefix, (-float("inf"), -float("inf")))
                    new_p_b = logsumexp(nb[0], logsumexp(p_b + p, p_nb + p))
                    next_beams[prefix] = (new_p_b, nb[1])
                    continue

                end = prefix[-1] if prefix else None
                new_prefix = prefix + (c,)

                if c == end:
                    # If repeated char, only transitions from blank add to non-blank of extended prefix
                    nb_ext = next_beams.get(new_prefix, (-float("inf"), -float("inf")))
                    new_p_nb_ext = logsumexp(nb_ext[1], p_b + p)
                    next_beams[new_prefix] = (nb_ext[0], new_p_nb_ext)

                    # Also allow staying on same prefix from non-blank (CTC "merge" case)
                    nb_same = next_beams.get(prefix, (-float("inf"), -float("inf")))
                    new_p_nb_same = logsumexp(nb_same[1], p_nb + p)
                    next_beams[prefix] = (nb_same[0], new_p_nb_same)
                else:
                    # Normal extension: can come from blank or non-blank
                    nb2 = next_beams.get(new_prefix, (-float("inf"), -float("inf")))
                    new_p_nb2 = logsumexp(nb2[1], logsumexp(p_b + p, p_nb + p))
                    next_beams[new_prefix] = (nb2[0], new_p_nb2)

        # Prune
        scored = []
        for prefix, (p_b, p_nb) in next_beams.items():
            scored.append((prefix, logsumexp(p_b, p_nb)))
        scored.sort(key=lambda x: x[1], reverse=True)

        beams = {}
        for prefix, _score in scored[:beam_width]:
            p_b, p_nb = next_beams[prefix]
            beams[prefix] = (p_b, p_nb)

    final = []
    for prefix, (p_b, p_nb) in beams.items():
        score = logsumexp(p_b, p_nb)
        # Convert token IDs -> string, and map '|' to space (torchaudio bundles typically use this)
        text = "".join(labels[i] for i in prefix).replace("|", " ").strip()
        final.append((text, score))

    final.sort(key=lambda x: x[1], reverse=True)

    # De-dup identical strings keeping best score
    dedup: dict[str, float] = {}
    for txt, sc in final:
        if txt and (txt not in dedup or sc > dedup[txt]):
            dedup[txt] = sc
    out = sorted(dedup.items(), key=lambda x: x[1], reverse=True)
    return out[:top_k]

best = ctc_beam_search_decode(
    emission=emission,
    labels=labels,
    blank_id=BLANK_ID,
    beam_width=50,
    top_k=3,
)
for i, (txt, score) in enumerate(best, start=1):
    print(f"Beam {i}: score={score:.2f}  text={txt}")



"""
Results:
Faster-Whisper:
[0.00s -> 4.18s]  You guys are killing my life!
[4.18s -> 10.50s]  You're the thing that slowly stops me, yeah!
[10.50s -> 17.00s]  I was made of all the things I have today!
[17.00s -> 24.00s]  Seasons never end, the worst is waking up!
[24.00s -> 28.00s]  Making someone, I'm not gonna make it!

WAV2VEC2_ASR_BASE_960H
No preprocessed waveform and vocals
THEY YOS A SHELL  A TNG TE SOLLY STOPSTY TOS MA INVALVETHE SINGS A AS AN ST IS A NO AWLSE TE TIN LAKE IN S MAN SAN AM NOT Y O LY GDS HE

Actual Lyrics:
I push my fingers into my eyes
It's the only thing that slowly stops the ache
But it's made of all the things I have to take
Jesus, it never ends, it works it's way inside
If the pain goes on, I'm not gonna make it

Results Discussion:

Current results show that, even when separating vocals from other sounds,
the current models have a poor detection and therefore we might want to focus 
more on similarities between sounds, such as beats, drums, etc. rather than lyrics 
as finding similarity across lyrics may prove to be difficult. 
"""

# So far, our encoder cannot find any elements due to the harshness of metal music.

