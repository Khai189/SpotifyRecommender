import torch
from torch import nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio import datasets
import torchaudio.transforms as transforms
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

torch.manual_seed(42)

# Experimentation, we start with an initial torchaudio model and build from there
waveform, sample_rate = torchaudio.load('test.mp3')
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
initial_model = bundle.get_model()

initial_model.eval()

expected_sample_rate = bundle.sample_rate
if sample_rate != expected_sample_rate:
    resampler = torchaudio.transforms.Resample(sample_rate, expected_sample_rate)
    waveform = resampler(waveform)

waveform = waveform.mean(dim=0, keepdim=True) if waveform.shape[0] > 1 else waveform
input_tensor = waveform.unsqueeze(0)

with torch.inference_mode():
    logits, _ = initial_model(waveform)
labels = bundle.get_labels()
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
    def __init__(self, labels, blank="|-"):  # Use a non-standard blank to avoid confusion in output
        super().__init__()
        self.labels = labels
        self.blank = blank
        self.blank_id = labels.index(blank) if blank in labels else len(labels)

    def forward(self, emission: torch.Tensor) -> str:
        indices = torch.argmax(emission, dim=-1)
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank_id]
        return "".join([self.labels[i] for i in indices])


decoder = GreedyCTCDecoder(labels=bundle.get_labels(), blank='-')

transcript = decoder(logits[0])

final_text = transcript.replace('|', ' ').strip()
print(f"Transcription: {final_text}")
