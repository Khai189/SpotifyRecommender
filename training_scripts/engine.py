import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder

device = "cuda" if torch.cuda.is_available() else "cpu"

