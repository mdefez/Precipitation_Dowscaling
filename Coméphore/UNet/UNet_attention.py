# The goal of this script is to implement a UNet class with temporal attention

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

class TemporalAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)  # optionnel mais conseillé

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        # Reshape pour appliquer l'attention temporelle sur chaque position (H, W)
        x_flat = x.permute(0, 3, 4, 1, 2).reshape(-1, T, C)  # (B*H*W, T, C)

        # Projeter en Q, K, V
        Q = self.q_proj(x_flat)
        K = self.k_proj(x_flat)
        V = self.v_proj(x_flat)

        # Attention scores
        attn_scores = (Q @ K.transpose(-2, -1)) / (C ** 0.5)  # (BHW, T, T)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Attention output
        out = attn_weights @ V  # (BHW, T, C)
        out = self.out_proj(out)  # (BHW, T, C)

        # Residual + normalisation
        out = self.norm(out + x_flat)  # (BHW, T, C)

        # Reshape vers (B, T, C, H, W)
        out = out.view(B, H, W, T, C).permute(0, 3, 4, 1, 2)

        return out  # (B, T, C, H, W)


class UNet_with_attention(nn.Module):
    def __init__(self,  hard_constraint_mass, temp_factor, spatial_factor, n_inputs, input_channels=2, base_channels=16):
        super().__init__()
        self.n_inputs = n_inputs
        self.temp_factor = temp_factor 
        self.spatial_factor = spatial_factor 
        self.hard_constraint_mass = hard_constraint_mass

        self.base_channels = base_channels
        self.attn1 = TemporalAttention(base_channels)
        self.attn2 = TemporalAttention(base_channels * 2)
        self.attn3 = TemporalAttention(base_channels * 4)
        self.attn4 = TemporalAttention(base_channels * 8)
        self.attn_bottleneck = TemporalAttention(base_channels * 16)

        self.attn_dec4 = TemporalAttention(base_channels * 16)
        self.attn_dec3 = TemporalAttention(base_channels * 8)
        self.attn_dec2 = TemporalAttention(base_channels * 4)
        self.attn_dec1 = TemporalAttention(base_channels * 2)

        # Shared convolution applied per timestep
        self.encoder1 = self.conv_block(input_channels, base_channels)
        self.encoder2 = self.conv_block(base_channels, base_channels * 2)
        self.encoder3 = self.conv_block(base_channels * 2, base_channels * 4)
        self.encoder4 = self.conv_block(base_channels * 4, base_channels * 8)

        self.pool = nn.MaxPool2d(2, 2)

        self.bottleneck = self.conv_block(base_channels * 8, base_channels * 16)

        self.upconv4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, 2)
        self.upconv3 = nn.ConvTranspose2d(base_channels * 16, base_channels * 4, 2, 2)
        self.upconv2 = nn.ConvTranspose2d(base_channels * 8, base_channels * 2, 2, 2)
        self.upconv1 = nn.ConvTranspose2d(base_channels * 4, base_channels, 2, 2)

        self.final_layer =  nn.Sequential(nn.Conv2d(self.base_channels * 2, out_channels=self.temp_factor, kernel_size=1),
                                          nn.ReLU())

    # To match size before concatenating
    def pad_to_match(self, x, ref):
        # Pad x spatially to match ref
        _, _, _, H, W = ref.shape
        _, _, _, h, w = x.shape
        dh = H - h
        dw = W - w
        if dh > 0 or dw > 0:
            x = F.pad(x, [0, dw, 0, dh])
        return x


    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
    
    # To apply to each aggregated frame (so that we keep the frame aspect thus we are allowed to compute attention)
    def apply_conv_per_t(self, x, conv):
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x = conv(x)
        _, C_out, H, W = x.shape
        x = x.view(B, T, C_out, H, W)
        return x
    
    def apply_pool_per_t(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x = self.pool(x)
        C, H, W = x.shape[1:]
        x = x.view(B, T, C, H, W)
        return x
    
    def upconv_per_t(self, x, upconv):
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x = upconv(x)
        C, H, W = x.shape[1:]
        x = x.view(B, T, C, H, W)
        return x
 
    def unet_forward(self, x): # Be careful, this is only the UNet block, the "real" forward is below
        # x: (B, T, C=2, H, W), we have two channels, the low res precip and the dem
        B, T, C, H, W = x.shape

        # Encoder
        x1 = self.apply_conv_per_t(x, self.encoder1)     
        #x1 = self.attn1(x1)
        p1 = self.apply_pool_per_t(x1)

        x2 = self.apply_conv_per_t(p1, self.encoder2)
        #x2 = self.attn2(x2)
        p2 = self.apply_pool_per_t(x2)

        x3 = self.apply_conv_per_t(p2, self.encoder3)
        #x3 = self.attn3(x3)
        p3 = self.apply_pool_per_t(x3)

        x4 = self.apply_conv_per_t(p3, self.encoder4)
        #x4 = self.attn4(x4)
        p4 = self.apply_pool_per_t(x4)

        # Bottleneck
        x5 = self.apply_conv_per_t(p4, self.bottleneck)
        x5 = self.attn_bottleneck(x5)

        x1 = x1[:, -1].unsqueeze(1)
        x2 = x2[:, -1].unsqueeze(1)
        x3 = x3[:, -1].unsqueeze(1)
        x4 = x4[:, -1].unsqueeze(1)
        x5 = x5[:, -1].unsqueeze(1) # Take only the last frame to decode, (B, 1, C, H, W)
        
        # Decoder
        d4 = self.upconv_per_t(x5, self.upconv4)
        d4 = self.pad_to_match(d4, x4)

        d4 = torch.cat([d4, x4], dim=2) # We concatenate on the channel dimension, which is the second one
        #d4 = self.attn_dec4(d4)

        d3 = self.upconv_per_t(d4, self.upconv3)
        d3 = self.pad_to_match(d3, x3)
        d3 = torch.cat([d3, x3], dim=2)
        #d3 = self.attn_dec3(d3)

        d2 = self.upconv_per_t(d3, self.upconv2)
        d2 = self.pad_to_match(d2, x2)
        d2 = torch.cat([d2, x2], dim=2)
        #d2 = self.attn_dec2(d2)

        d1 = self.upconv_per_t(d2, self.upconv1)
        d1 = self.pad_to_match(d1, x1)
        d1 = torch.cat([d1, x1], dim=2)
        #d1 = self.attn_dec1(d1)             # (B, T, C', H, W)

        d1 = d1[:, -1]
        out = self.final_layer(d1)  # (B, temp_factor, H, W)

        return out
    

    def forward(self, frames, dem, apply_constraint = True): # frames is a list of coarse inputs, dem is the dem associated to the tile

        # frames = [frame_0, ..., frame_-1] where frame = (B, 1, 1, H, W) & dem = (B, 1, 100, 100)

        ### FIRST STEP
        # Before anything, if the last image (we seek to downsample) is all zeroes, then we force the predictions to be zero. Otherwise, we pass it to the unet

        # Create all outputs to zeros
        batch_size = frames[-1].size(0)

        outputs = torch.zeros(batch_size, self.temp_factor, 100, 100, device=frames[-1].device)  
        
        # Check if for some sample, the two last input images are all zeros
        all_zero_inp0 = torch.all(frames[-1] == 0, dim=(-2, -1))  # [B]

        non_null_mask = ~(all_zero_inp0).squeeze()  # [B], True if not entirely 0

        ### SECOND STEP
        # If not 0, pass them to the UNet. The UNet itself never sees any all zeroes frames
        if non_null_mask.any():
            # We only pass the non zero samples
            ### ATTENTION ENLEVER LES SQUEEZE(0) POUR LE TRAIN
            frames_non_null = [frame[non_null_mask] for frame in frames] # List of non null frame, now frame = (B', 1, 1, H, W) where B' is the number of non null samples
            dem_non_null = dem[non_null_mask]

            # Upsample the 2 frames with bicubic interpolation
            B, T, C, H, W = frames_non_null[0].shape
            # Change the shape to feed the interpolater
            frames_up = [frame.view(B * T, C, H, W) for frame in frames_non_null] 
            frames_up = [F.interpolate(frame, size=(100, 100), mode='bicubic', align_corners=False) for frame in frames_up] 
            frames_up = [frame.view(B, T, C, 100, 100) for frame in frames_up]
            

            # Concat the non zeros sample to build a new "batch", with fewer samples than the inital ones (we made the all zeros go away)
            x_non_null = torch.cat(frames_up, dim=1)  # [B', T, 1, H, W] where T is the number of aggregated frames

            # Final input
            dem_expanded = dem_non_null.unsqueeze(1).expand(-1, self.n_inputs, -1, -1, -1) # dem is now [B', T, 1, H, W]
            input = torch.cat((x_non_null, dem_expanded), dim=2) # input is [B', T, 2, H, W]

            # Pass it to the UNet
            output_non_null = self.unet_forward(input) # output is [B', 6, H, W]

            ### THIRD STEP
            # Here we apply (if asked) the hard constraint mass strategy
            # The constraint is such that the mass should be the same in the last aggregated frame and the predictions
            if apply_constraint == True:

                if self.hard_constraint_mass != None:
                    if self.hard_constraint_mass == "additive":
                        # Choose the mass reference. It should be low res (time & space). Here we take the last input 
                        P_LR = frames_non_null[-1].sum(dim=(-2, -1), keepdim = True) * self.temp_factor  # Shape (B', 1, 1, 1, 1)
                        P_LR = P_LR.squeeze(4) # Shape (B', 1, 1, 1)

                        # Compute the sum at the denominator
                        sum = output_non_null.sum(dim=(1, 2, 3), keepdim=True) / (self.spatial_factor ** 2)  # shape: (B', 1, 1, 1)


                        # Compute the final (constrained) outputs
                        output_final = output_non_null + (P_LR - sum) * ((self.spatial_factor / 100)**2) / self.temp_factor  # shape: (B', 6, 100, 100)
                        output_non_null = output_final                      # Rename it

                    else: # This is thus the multiplicative strategy
                        strategy, f = self.hard_constraint_mass

                        f_output = f(output_non_null)  # shape: (B', temp_factor, 100, 100)

                        # Choose the mass reference. It should be low res (time & space) 
                        P_LR = frames_non_null[-1].sum(dim=(-2, -1), keepdim = True) * self.temp_factor   # Shape (B', 1, 1, 1, 1)
                        P_LR = P_LR.squeeze(4) # Shape (B', 1, 1, 1)

                        # Compute the sum at the denominator
                        sum_f = f_output.sum(dim=(1, 2, 3), keepdim=True) / (self.spatial_factor ** 2)  # shape: (B', 1, 1, 1)


                        # Compute the final (constrained) outputs
                        output_final = f_output * (P_LR / sum_f)   # shape: (B', 6, 100, 100)
                        output_non_null = output_final                      # Rename it

                # Setting the outputs
                outputs[non_null_mask.squeeze()] = output_non_null  # [B, 6, H, W]


        else: # If all the batch is all zero (it might happened when the batch_size is low), we have to force the tensor to allow gradient computing
            outputs = torch.zeros(batch_size, self.temp_factor, 100, 100, device=frames[-1].device, requires_grad = True)  


        return outputs  









