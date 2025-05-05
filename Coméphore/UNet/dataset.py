# The goal of this script is to develop a dataset class to feed the dataloader
# Each dataset is defined for a specific tile
# The class should have a __getitem__ method that return an input & target to the model : 
#   - n following low res frames on the same tile, where n can be chosen
#   - the tile's DEM 
#   - the 6 targets corresponding to the last low res frame

import os
import numpy as np
import torch
from torch.utils.data import Dataset

class RainSuperResDataset(Dataset):
    def __init__(self, input_root, output_root, channel_root, hor, vert, temp_factor, spatial_factor, train=True, n_days = 5, n_inputs = 1, delta = False): # Channel refers to the DEM
        
        self.samples = [] # This is a list of all the inputs

        self.temp_factor = temp_factor
        self.spatial_factor = spatial_factor

        self.delta = delta # If we want to predict the delta instead of the real frames (except for the first one)

        # Folder where the data is stored
        self.input_root = input_root
        self.output_root = output_root
        self.channel_root = channel_root
        self.hor = hor      # Coordinates of the horizontal tile
        self.vert = vert    # Coordinates of the vertical tile
        self.domain = f"tile_hor_{self.hor}_vert_{self.vert}"     # Name of the domain, given the coordinates        

        # We train on 2023 & test on 2024
        self.n_days = n_days # We only use the n-th first days of each month
        if train:
            years = ['2023'] 
        else:
            years = ['2024']


        # Add all the inputs to the list
        for year in years:
            input_files = sorted(os.listdir(os.path.join(input_root, year, self.domain))) # Select the right tile
            input_files = [file for file in input_files if int(file[16:18]) < self.n_days] # We only use the n-th first days of each month
            input_times = [int(f[10:20]) for f in input_files] # Get timesteps from the filename
            input_times.sort()

            # We select n consecutive low res frames
            for i in range(len(input_times) - (n_inputs - 1)): 
                t0 = input_times[i]
                following_frames = [t0]
                add_the_sample = True
                for k in range(1, n_inputs): # We loop until n - 1 to have exaclty n inputs
                    t_next = input_times[i + k]
                    following_frames.append(t_next)

                for t in following_frames:  # Some exceptions should be handled + we should not overlap on the next day
                    if (t % 10 == 1 and self.temp_factor == 6) or ((t%100) + self.temp_factor >= 24):
                        add_the_sample = False
                
                
                if add_the_sample == True: # All following frames are temp_factor hours away and the sample should be add
                    dict_to_add = {"year": year,
                            "domain": self.domain,   # DEM
                    }
                    dict_to_add["low_res_idx"] = following_frames

                    self.samples.append(dict_to_add)

    def __len__(self):
        return len(self.samples)
    
    def input_format(self, timestep): # Return the correct input filename corresponding to the timestep
        return f"beggining_{timestep}_temp_factor_{self.temp_factor}_spatial_factor_{self.spatial_factor}.npy"
    
    def target_format(self, timestep): # Return the correct output filename corresponding to the timestep
        return f"{timestep}.npy"
    
    def dem_name(self, domain): # Return the correct dem filename corresponding to the domain
        hor = domain[9]
        vert = domain[16]
        return f"dem_hor_{hor}_vert_{vert}.npy"

    def __getitem__(self, idx): # Return (list of low frames, dem, target), basically (input, target)
        sample = self.samples[idx]
        year = sample["year"]
        domain = sample["domain"]
        low_res_idx = sample["low_res_idx"]

        # Load the low res inputs
        low_res_frames = [np.load(os.path.join(self.input_root, year, domain, self.input_format(t))) for t in low_res_idx]
        # Transform into tensors
        low_res_tensors = [torch.tensor(npy).unsqueeze(0).unsqueeze(0).float() for npy in low_res_frames] # List of (1, 1, H, W)

        # Load the DEM
        channel = np.load(os.path.join(self.channel_root, self.dem_name(domain)))
        channel = torch.tensor(channel).unsqueeze(0).float() # (1, H, W)

        # Load the high res targets
        targets = []

        # We have to select the n targets corresponding to the last low res frame
        count = 0
        for t in range(low_res_idx[-1], low_res_idx[-1] + self.temp_factor):  
            target_path = os.path.join(self.output_root, year, domain, self.target_format(t))

            if self.delta == True:                          # The first target should be the real frame, the following should be deltas
                if count == 0:                              # First frame
                    target = np.load(target_path)
                    target_ref = target.copy()

                if count >= 1:                              # Following frames
                    target = np.load(target_path)
                    target = target - target_ref            # t_n - t_0

            else:                                           # If delta == False, simply add the real frames
                target = np.load(target_path)

            targets.append(torch.tensor(target).unsqueeze(0).float())
            count += 1
            
        targets = torch.stack(targets) 
        targets = targets.squeeze(1) # (temp_factor, H, W)

        return low_res_tensors, channel, targets, low_res_idx # (List of inputs, dem, Tensor of targets, List of timesteps corresponding to targets)
