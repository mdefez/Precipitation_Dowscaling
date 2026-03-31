from Coméphore.Config import working_directory, data_directory


import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RainSuperResDataset(Dataset):
    def __init__(self, input_root, output_root, hor, vert, temp_factor, spatial_factor, train=True, n_days = 5):
        
        self.samples = [] # This is a list of all the inputs

        self.temp_factor = temp_factor
        self.spatial_factor = spatial_factor

        # Folder where the data is stored
        self.input_root = input_root
        self.output_root = output_root
        self.hor = hor      # Coordinates of the horizontal tile
        self.vert = vert    # Coordinates of the vertical tile
        self.domain = f"tile_hor_{self.hor}_vert_{self.vert}"     # Name of the domain, given the coordinates        

        # We train on 2023 & test on 2024
        self.n_days = n_days # We only use the n-th first days of each month
        if train:
            self.year = '2023'
        else:
            self.year = '2024'

        self.to_tensor = T.ToTensor()

        self.lr_path = os.path.join(self.input_root, self.year, self.domain)
        self.hr_path = os.path.join(self.output_root, self.year, self.domain)

        self.files_lr = sorted([f for f in os.listdir(self.lr_path)])
        self.files_hr = sorted([f for f in os.listdir(self.hr_path)])

    def __len__(self):
        return len(self.files_lr)

    def __getitem__(self, idx):

        lr_name = self.files_lr[idx]
        hr_name = self.files_hr[idx]

        lr = np.load(os.path.join(self.lr_path, lr_name)).astype(np.float32)
        hr = np.load(os.path.join(self.hr_path, hr_name)).astype(np.float32)

        # ajouter channel dimension
        lr = lr[None, :, :]   # (H,W) → (1,H,W)
        hr = hr[None, :, :]

        return lr, hr