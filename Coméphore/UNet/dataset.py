# The goal of this script is to develop a dataset class to feed the dataloader
# Each dataset is defined for a specific tile
# The class should have a __getitem__ method that return an input & output to the model : 
#   - 2 (following) low res frames on the same tile
#   - the tile's DEM 
#   - the 6 correspondings targets

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class RainSuperResDataset(Dataset):
    def __init__(self, input_root, output_root, channel_root, hor, vert, train=True): # Channel refers to the DEM
        self.samples = [] # This is a list of all the inputs

        # Folder where the data is stored
        self.input_root = input_root
        self.output_root = output_root
        self.channel_root = channel_root
        self.hor = hor      # Coordinates of the horizontal tile
        self.vert = vert    # Coordinates of the vertical tile
        self.domain = f"tile_hor_{self.hor}_vert_{self.vert}"     # Name of the domain, given the coordinates        

        # We train on 2023 & test on 2024
        if train:
            years = ['2023'] 
        else:
            years = ['2024']


        # Add all the inputs to the list
        for year in years:
            input_files = sorted(os.listdir(os.path.join(input_root, year, self.domain))) # Select the right tile
            input_times = [int(f[10:20]) for f in input_files] # Get timesteps from the filename
            input_times.sort()

            # We select consecutive images if there is a 6 hour gap 
            for i in range(len(input_times) - 1): 
                t0 = input_times[i]
                t1 = input_times[i + 1]
                if t1 - t0 == 6:  # Add the couple to the list of samples
                    self.samples.append({
                        "year": year,
                        "domain": self.domain,   # DEM
                        "t0": t0,           # First frame
                        "t1": t1            # Last frame
                    })

    def __len__(self):
        return len(self.samples)
    
    def input_format(self, timestep): # Return the correct input filename corresponding to the timestep
        return f"beggining_{timestep}_temp_factor_6_spatial_factor_25.npy"
    
    def target_format(self, timestep): # Return the correct output filename corresponding to the timestep
        return f"{timestep}.npy"
    
    def dem_name(self, domain): # Return the correct dem filename corresponding to the domain
        hor = domain[9]
        vert = domain[16]
        return f"dem_hor_{hor}_vert_{vert}.npy"

    def __getitem__(self, idx): # Return (image_timestep_t, image_timestep_t+1, dem, target), basically (input, target)
        sample = self.samples[idx]
        year = sample["year"]
        domain = sample["domain"]
        t0 = sample["t0"]
        t1 = sample["t1"]

        # Load the low res inputs
        inp0 = np.load(os.path.join(self.input_root, year, domain, self.input_format(t0)))
        inp1 = np.load(os.path.join(self.input_root, year, domain, self.input_format(t1)))

        inp0 = torch.tensor(inp0).unsqueeze(0).unsqueeze(0).float()
        inp1 = torch.tensor(inp1).unsqueeze(0).unsqueeze(0).float()

        # Load the DEM
        channel = np.load(os.path.join(self.channel_root, self.dem_name(domain)))
        channel = torch.tensor(channel).unsqueeze(0).float() 

        # Load the high res targets
        targets = []

        # We have to select 6 targets between t0 and t1 + 5, different strategies are possible because there are 12 images
        def strategy(t0, t1): # Selects the index corresponding to the time steps
            strat_1 = range(t0, t1) 
            strat_2 = range(t0 + 3, t1 + 3)

            return strat_2

        for t in strategy(t0, t1):  
            target_path = os.path.join(self.output_root, year, domain, self.target_format(t))
            target = np.load(target_path)
            targets.append(torch.tensor(target).unsqueeze(0).float())
        targets = torch.stack(targets) 
        targets = targets.squeeze(1)

        return inp0.squeeze(0), inp1.squeeze(0), channel, targets
