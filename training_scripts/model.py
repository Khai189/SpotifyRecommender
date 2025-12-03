import torch
from torch import nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio import datasets, transforms
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

class SongAnalyzer(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_units: int,
                 output_units: int):
        super().__init__()

