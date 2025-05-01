# This scripts implements (untrainable) baselines such as kNN & bicubic interpolation
# Given the nature of the super-resolving methods, the latter can only perform it in space and not in time
# Thus, if the temporal factor is greater than one, all the sub images will be the same

import torch
import torch.nn as nn
import torch.nn.functional as F

class bicubic(nn.Module):
    def __init__(self, temp_factor, spatial_factor):
        super().__init__()
        
        self.temp_factor = temp_factor 
        self.spatial_factor = spatial_factor 

    def forward(self, frames, dem, apply_constraint = False): # We won't use the dem
        last_frame = frames[-1].squeeze(1) # We super resolve only the last image. We should provide the interpolate function in shape (B, C = 1, H, W)

        # Super resolve in space
        last_frame_sr = F.interpolate(last_frame, size=(100, 100), mode='bicubic')

        # "Super resolve in time"
        frames_sr = [last_frame_sr for k in range(self.temp_factor)]
        frames_sr = torch.stack(frames_sr, dim=0).squeeze(0) # Concatenate the frames into a (self.temp_factor, H, W) tensor

        return frames_sr
    

class nearest_neighbor(nn.Module):
    def __init__(self, temp_factor, spatial_factor):
        super().__init__()
        
        self.temp_factor = temp_factor 
        self.spatial_factor = spatial_factor 

    def forward(self, frames, dem, apply_constraint = False): # We won't use the dem
        last_frame = frames[-1].squeeze(1) # We super resolve only the last image. We should provide the interpolate function in shape (B, C = 1, H, W)

        # Super resolve in space
        last_frame_sr = F.interpolate(last_frame, size=(100, 100), mode='nearest')

        # "Super resolve in time"
        frames_sr = [last_frame_sr for k in range(self.temp_factor)]
        frames_sr = torch.stack(frames_sr, dim=0).squeeze(0) # Concatenate the frames into a (self.temp_factor, H, W) tensor

        return frames_sr
