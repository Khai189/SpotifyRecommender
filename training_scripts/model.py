import torch
from torch import nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio import datasets, transforms
device = "cuda" if torch.cuda.is_available() else "cpu"

class SongAnalyzer(nn.Module):
    def __init__(self,
                 ):
        super().__init__()

