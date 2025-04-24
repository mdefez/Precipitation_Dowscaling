# The goal of this script is to implement a UNet class with temporal attention

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

# Self attention : (B, T, C, H, W) to (B, T, C, H, W)
class SelfTemporalAttention(nn.Module):
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

# Cross attention (focus on the last frame) : (B, T, C, H, W) to (B, 1, C, H, W)
class CrossTemporalAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, embed_dim)
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        assert T >= 2, "Need at least 2 frames for cross-attention"

        # Last frame (query) — shape (B, 1, C, H, W)
        x_q = x[:, -1:]
        # All previous frames (key/value) — shape (B, T-1, C, H, W)
        x_kv = x[:, :-1]

        # Reshape to apply attention at each pixel location independently
        q = x_q.permute(0, 3, 4, 1, 2).reshape(-1, 1, C)      # (B*H*W, 1, C)
        kv = x_kv.permute(0, 3, 4, 1, 2).reshape(-1, T-1, C)  # (B*H*W, T-1, C)

        # Project to Q, K, V
        Q = self.q_proj(q)   # (BHW, 1, embed_dim)
        K = self.k_proj(kv)  # (BHW, T-1, embed_dim)
        V = self.v_proj(kv)  # (BHW, T-1, embed_dim)

        # Scaled dot-product attention
        attn_scores = (Q @ K.transpose(-2, -1)) / (self.embed_dim ** 0.5)  # (BHW, 1, T-1)
        attn_weights = torch.softmax(attn_scores, dim=-1)                   # (BHW, 1, T-1)
        out = attn_weights @ V                                              # (BHW, 1, embed_dim)

        # Project back and residual + norm
        out = self.out_proj(out)                  # (BHW, 1, C)
        out = self.norm(out + q)                  # (BHW, 1, C)

        # Reshape back to (B, 1, C, H, W)
        out = out.view(B, H, W, 1, C).permute(0, 3, 4, 1, 2)

        return out  # (B, 1, C, H, W)


class UNet_with_attention(nn.Module):
    def __init__(self, temp_factor, spatial_factor, model_parameters, input_channels=2, base_channels=16):
        super().__init__()
        
        self.temp_factor = temp_factor 
        self.spatial_factor = spatial_factor 
        self.hard_constraint_mass = model_parameters[0]
        self.n_inputs = model_parameters[1]
        self.attention = model_parameters[2]          # Attention strategy

        self.base_channels = base_channels
        self.attn1 = SelfTemporalAttention(base_channels)
        self.attn2 = SelfTemporalAttention(base_channels * 2)
        self.attn3 = SelfTemporalAttention(base_channels * 4)
        self.attn4 = SelfTemporalAttention(base_channels * 8)
        self.attn_bottleneck = CrossTemporalAttention(base_channels * 16)


        # Encoder
        # Shared convolution applied per timestep
        self.encoder1 = self.conv_block(input_channels, base_channels)
        self.encoder2 = self.conv_block(base_channels, base_channels * 2)
        self.encoder3 = self.conv_block(base_channels * 2, base_channels * 4)
        self.encoder4 = self.conv_block(base_channels * 4, base_channels * 8)

        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = self.conv_block(base_channels * 8, base_channels * 16)

        # Decoder
        # At each step, we upsampled, skip connection and add a conv block (double convolution)

        # We can use transposed convolution but it is known to generate artifacts in blocks
        self.transconv4 = self.upconv(base_channels * 16, base_channels * 8, 2, 2)
        self.transconv3 = self.upconv(base_channels * 8, base_channels * 4, 2, 2)
        self.transconv2 = self.upconv(base_channels * 4, base_channels * 2, 2, 2)
        self.transconv1 = self.upconv(base_channels * 2, base_channels, 2, 2)

        # Or we can use upsampling
        # To fill

        # Or pixel shuffling
        # To fill

        # Conv blocks in the decoder
        self.decoder4 = self.conv_block(base_channels * 16, base_channels * 8)
        self.decoder3 = self.conv_block(base_channels * 8, base_channels * 4)
        self.decoder2 = self.conv_block(base_channels * 4, base_channels * 2)
        self.decoder1 = self.conv_block(base_channels * 2, base_channels)



        # Store the layers to make the forward more readable (and iterate over it)
        self.encoders = nn.ModuleList([self.encoder1, self.encoder2, self.encoder3, self.encoder4])
        self.attention_blocks = nn.ModuleList([self.attn1, self.attn2, self.attn3, self.attn4])

        self.upconvs  = nn.ModuleList([self.transconv4, self.transconv3, self.transconv2, self.transconv1])
        self.decoders = nn.ModuleList([self.decoder4, self.decoder3, self.decoder2, self.decoder1])


        # Final layer, 1*1 convolution
        self.final_layer =  nn.Sequential(nn.Conv2d(self.base_channels, out_channels=self.temp_factor, kernel_size=1),
                                          nn.ReLU())

    # To match x's size to target before concatenating
    def pad_to_match(self, x, target):
        diff_y = target.size(2) - x.size(2)
        diff_x = target.size(3) - x.size(3)

        pad_left = diff_x // 2
        pad_right = diff_x - pad_left
        pad_top = diff_y // 2
        pad_bottom = diff_y - pad_top

        x_padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate') # We can use either "replicate" or "reflect" 
        return x_padded


    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
    
    def upconv(self, in_channels, out_channels, stride=2, padding = 2):
        return nn.ConvTranspose2d(in_channels, out_channels, stride, padding)
    

    def upsample(self, scale_factor=2):
        return nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        

    # To apply to each aggregated frame (so that we keep the frame aspect thus we are allowed to compute temporal attention)
    # One can also apply those functions to a sequence of one image (without computing attention)
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
    

 
    def unet_forward(self, x): # Be careful, this is only the UNet block, the "real" forward is below
        # x: (B, T, C=2, H, W), we have two channels, the low res precip and the dem. T is the number of inputs

        # ===== Encoder =====
        encoder_outputs = []
        x_in = x
        for k in range(len(self.encoders)):
            x_out = self.apply_conv_per_t(x_in, self.encoders[k])
            encoder_outputs.append(x_out)
            if self.attention == "encoder":                     # Compute attention depending on the strategy
                x_out = self.attention_blocks[k](x_out)        # We compute attention after saving the soon-to-be skip connected
            x_in = self.apply_pool_per_t(x_out)

        # Bottleneck
        x_bottleneck = self.apply_conv_per_t(x_in, self.bottleneck)
        if self.attention in ["encoder", "bottleneck"]:
            x_bottleneck = self.attn_bottleneck(x_bottleneck)

        # Keep only the low res frame to decode given that we don't compute attention in the decoder
        # This goes from (B, T, C, H, W) to (B, C, H, W)
        encoder_outputs = [f[:, -1] for f in encoder_outputs]
        x_bottleneck = x_bottleneck[:, -1]


        # Decoder
        x = x_bottleneck
        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            skip = encoder_outputs[-(i+1)]  
            x = self.pad_to_match(x, skip)
            x = torch.cat([x, skip], dim=1)
            x = self.decoders[i](x)

        # Final output
        out = self.final_layer(x)

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









