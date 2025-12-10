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


audio, rate = torchaudio.load('test.mp3')

if rate != 44100:
    audio = torchaudio.transforms.Resample(rate, 44100)(audio)

audio_tensor = audio.float().unsqueeze(0)
separator = torch.hub.load('sigsep/open-unmix-pytorch', 'umxhq', device="cpu")

with torch.inference_mode():
    estimates = separator(audio_tensor)[0]


stems = ['vocals', 'drums', 'bass', 'other']
output_path = 'umx_output'
os.makedirs(output_path, exist_ok=True)

for i, stem in enumerate(stems):
    output_file = os.path.join(output_path, f'umx_{stem}.wav')
    torchaudio.save(output_file, estimates[i].cpu(), 44100)



torch.manual_seed(42)

# Experimentation, we start with an initial torchaudio model and build from there.
# We'll use faster whisper, which is an implementation of ChatGPT's whisper, and hopefully build
# our own custom model if performance sees fit

test_vocal_path = os.path.join(output_path, "umx_vocals.wav")

print("Starting model download...")
model_size = "large-v3"

model = WhisperModel(model_size,
                     device="cpu",
                     compute_type="int8"
                     )

segments, info = model.transcribe("test.mp3", beam_size=5)

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
BLANK_ID = len(labels) - 1

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


# Example greedy decoder taken from PyTorch
class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank_index):  # Use a non-standard blank to avoid confusion in output
        super().__init__()
        self.labels = labels
        self.blank_id = blank_index

    def forward(self, emission: torch.Tensor) -> str:
        indices = torch.argmax(emission, dim=-1)
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank_id]
        return "".join([self.labels[i] for i in indices])


decoder = GreedyCTCDecoder(labels=bundle.get_labels(), blank_index=0)

transcript = decoder(logits[0])

final_text = transcript.replace('|', ' ').strip()
print(f"Transcription: {final_text}")

# So far, our encoder cannot find any elements due to the harshness of metal music.

