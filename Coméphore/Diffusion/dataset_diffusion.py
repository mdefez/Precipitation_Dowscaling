# For the moment it's a dummy dataset to debug

import torch
from torch.utils.data import Dataset

class DummySequenceDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=5, channels=1, height=32, width=32):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.channels = channels
        self.height = height
        self.width = width

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Générer A et C comme des séquences de bruit ~ N(0, 1)
        A_seq = torch.randn(self.seq_len, self.channels, self.height, self.width)
        C_seq = torch.randn(self.seq_len, self.channels, self.height, self.width)

        # Générer un résidu propre (R), ici aussi un bruit normal
        R = torch.randn(self.channels, self.height, self.width)

        # Dernière frame de C
        C_T = C_seq[-1]

        # Target B = C_T + R
        B = C_T + R

        return A_seq, C_seq, B
