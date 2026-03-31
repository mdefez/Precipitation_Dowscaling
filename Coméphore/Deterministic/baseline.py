# This scripts implements (untrainable) baselines such as kNN & bicubic interpolation
# Given the nature of the super-resolving strategy (we SR only the last frame), the model can only perform it in space and not in time
# Thus, if the temporal factor is greater than one, all the sub images will be the same

import torch
import torch.nn as nn
import torch.nn.functional as F

# Input is a list of length = n_inputs and item = (B, 1, 1, H, W) low res
# Output is (B, self.temp_factor, H, W) high res
class bicubic(nn.Module):
    def __init__(self, temp_factor, spatial_factor):
        super().__init__()
        
        self.temp_factor = temp_factor 
        self.spatial_factor = spatial_factor 

    def forward(self, frames, dem, apply_constraint = False): # We won't use the dem. Frames is a list of length = n_inputs and item = (B, 1, 1, H, W)
        last_frame = frames[-1].squeeze(1) # We super resolve only the last image. We should provide the interpolate function in shape (B, C = 1, H, W)

        # Super resolve in space
        last_frame_sr = F.interpolate(last_frame, size=(100, 100), mode='bicubic').squeeze(1) # (B, H, W)

        # "Super resolve in time"
        frames_sr = [last_frame_sr for k in range(self.temp_factor)]        # List of lenght self.temp_facotr, item = (B, H, W)
        frames_sr = torch.stack(frames_sr, dim=0)       # Concatenate the frames into a (self.temp_factor, B, H, W) tensor
        frames_sr = frames_sr.permute(1, 0, 2, 3)       # (B, self.temp_factor, H, W) 

        # Make it non-negative
        frames_sr = torch.clamp(frames_sr, min=0)

        return frames_sr
    
# Input is a list of length = n_inputs and item = (B, 1, 1, H, W) low res
# Output is (B, self.temp_factor, H, W) high res
class nearest_neighbor(nn.Module):
    def __init__(self, temp_factor, spatial_factor):
        super().__init__()
        
        self.temp_factor = temp_factor 
        self.spatial_factor = spatial_factor 

    def forward(self, frames, dem, apply_constraint = False): # We won't use the dem. Frames is a list of length = n_inputs and item = (B, 1, 1, H, W)
        last_frame = frames[-1].squeeze(1) # We super resolve only the last image. We should provide the interpolate function in shape (B, C = 1, H, W)

        # Super resolve in space
        last_frame_sr = F.interpolate(last_frame, size=(100, 100), mode='nearest').squeeze(1) # (B, H, W)

        # "Super resolve in time"
        frames_sr = [last_frame_sr for k in range(self.temp_factor)]        # List of lenght self.temp_facotr, item = (B, H, W)
        frames_sr = torch.stack(frames_sr, dim=0)       # Concatenate the frames into a (self.temp_factor, B, H, W) tensor
        frames_sr = frames_sr.permute(1, 0, 2, 3)       # (B, self.temp_factor, H, W) 

        return frames_sr
    

class edsr_baseline(nn.Module):
    def __init__(self, temp_factor, spatial_factor, models, list_scales):
        super().__init__()
        
        self.temp_factor = temp_factor 
        self.spatial_factor = spatial_factor 

        self.models = models
        self.list_scales = list_scales

        self.p = 1
        for scale in self.list_scales:
            self.p *= scale


    def forward(self, frames, dem, apply_constraint = False): # We won't use the dem. Frames is a list of length = n_inputs and item = (B, 1, 1, H, W)
        last_frame = frames[-1].squeeze(1) # We super resolve only the last image. We should provide the interpolate function in shape (B, C = 1, H, W)

        # Super resolve in space
        for model in self.models:
            last_frame = model(last_frame)

        last_frame_sr = F.interpolate(last_frame, scale_factor = self.spatial_factor/self.p, mode='nearest') # (B, C, H, W)

        return last_frame_sr

