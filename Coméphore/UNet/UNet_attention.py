# The goal of this script is to implement a UNet class with temporal attention

import torch
import torch.nn as nn
import torch.nn.functional as F


# Self attention : (B, T, C, H, W) to (B, T, C, H, W)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Takes (B, T, C, H, W) as input and returns same size by passing it to a temporal attention mecanism
# Every pixel of the last frame will be transformed according to its previous values (from the other frames)
class TemporalMultiHeadAttention(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.permute(0, 3, 4, 1, 2).contiguous()  # (B, H, W, T, C)
        x = x.view(B * H * W, T, C)  # (B*H*W, T, C), we compute attention for each B*H*W (pixel)

        Q = self.q_proj(x)  # (B*H*W, T, C)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Multiheads
        Q = Q.view(-1, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B*H*W, heads, T, head_dim)
        K = K.view(-1, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(-1, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B*H*W, heads, T, T)
        attn_weights = F.softmax(attn_scores, dim=-1) # Softmax on the columns
        attn_output = torch.matmul(attn_weights, V)  # (B*H*W, heads, T, head_dim)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B * H * W, T, C)  # concat heads
        out = self.out_proj(attn_output)  # (B*H*W, T, C)

        out = out.view(B, H, W, T, C).permute(0, 3, 4, 1, 2).contiguous()  # (B, T, C, H, W)
        return out

# Usual transformers block that performs LayerNorm --> Temporal Attention --> LayerNorm --> MLP
class TemporalTransformerBlock(nn.Module):
    def __init__(self, channels, num_heads = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = TemporalMultiHeadAttention(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape

        # --- Attention block ---
        x_reshaped = x.permute(0, 3, 4, 1, 2).contiguous().view(B * H * W, T, C) # (B * H * W, T, C)
        x_norm = self.norm1(x_reshaped)
        x_norm = x_norm.view(B, H, W, T, C).permute(0, 3, 4, 1, 2)  # (B, T, C, H, W)
        attn_out = self.attn(x_norm)
        x = x + attn_out  # Residual

        # --- MLP block ---
        x_reshaped = x.permute(0, 3, 4, 1, 2).contiguous().view(B * H * W, T, C)
        x_norm = self.norm2(x_reshaped)
        mlp_out = self.mlp(x_norm)
        mlp_out = mlp_out.view(B, H, W, T, C).permute(0, 3, 4, 1, 2)  # (B, T, C, H, W)
        x = x + mlp_out  # Residual

        return x

# Takes (B, T, C, H, W) as input and returns same size by passing it to a spatial attention mecanism
# Every pixel of the LAST FRAME will be transformed according to its neighbor values (from the same last frame)
class LocalSpatialAttention(nn.Module):
    def __init__(self, channels, num_heads, window_size=3):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, x):
        B, T, C, H, W = x.shape

        last_image = x.clone() 
        last_image = last_image[:, -1]  # Take last frame: (B, C, H, W)

        # Unfold to get local windows for each pixel
        unfold = nn.Unfold(kernel_size=self.window_size, padding=self.window_size // 2)
        patches = unfold(last_image)  # (B, C * n², H*W)
        patches = patches.transpose(1, 2).view(B, H*W, self.window_size**2, C)  # (B, H*W, n², C)

        # Project Q, K, V
        Q = self.q_proj(patches)
        K = self.k_proj(patches)
        V = self.v_proj(patches)

        # Split heads
        Q = Q.view(B, H*W, self.window_size**2, self.num_heads, self.head_dim).transpose(2, 3)  # (B, H*W, heads, n², head_dim)
        K = K.view(B, H*W, self.window_size**2, self.num_heads, self.head_dim).transpose(2, 3)
        V = V.view(B, H*W, self.window_size**2, self.num_heads, self.head_dim).transpose(2, 3)

        # Attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H*W, heads, n², n²)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # (B, H*W, heads, n², head_dim)

        # Collapse heads
        attn_output = attn_output.transpose(2, 3).reshape(B, H*W, self.window_size**2, C)

        # Take the center token (position in the patch)
        center_index = self.window_size**2 // 2
        output = attn_output[:, :, center_index, :]  # (B, H*W, C)

        # Restore image shape
        output = output.transpose(1, 2).view(B, C, H, W)  # (B, C, H, W)

        # Returns the whole temporal sequence with the just modified last image
        x[:, -1] = output

        return x  # (B, T, C, H, W)

# Usual transformers block that performs LayerNorm --> Spatial Attention --> LayerNorm --> MLP
class SpatialTransformerBlock(nn.Module):
    def __init__(self, channels, window_size=3, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = LocalSpatialAttention(channels, num_heads, window_size)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape

        # --- Attention block ---
        x_reshaped = x.permute(0, 3, 4, 1, 2).contiguous().view(B * H * W, T, C)
        x_norm = self.norm1(x_reshaped)
        x_norm = x_norm.view(B, H, W, T, C).permute(0, 3, 4, 1, 2)  # (B, T, C, H, W)
        
        # Compute spatial attention only on the last frame of the sequence
        x_out = self.attn(x_norm)  # (B, T, C, H, W)

        # --- MLP block ---
        x_reshaped = x_out.permute(0, 3, 4, 1, 2).contiguous().view(B * H * W, T, C)
        x_norm = self.norm2(x_reshaped)
        mlp_out = self.mlp(x_norm)
        mlp_out = mlp_out.view(B, H, W, T, C).permute(0, 3, 4, 1, 2)  # (B, T, C, H, W)
        x_out = x_out + mlp_out  # Residual connection

        return x_out










class UNet_with_attention(nn.Module):
    def __init__(self, temp_factor, spatial_factor, model_parameters, input_channels=2, base_channels=16):
        super().__init__()
        
        self.temp_factor = temp_factor 
        self.spatial_factor = spatial_factor 
        self.hard_constraint_mass = model_parameters[0]
        self.n_inputs = model_parameters[1]
        self.attention = model_parameters[2]          # Attention strategy

        self.base_channels = base_channels

        # Temporal transformers
        self.temp_attn1 = TemporalTransformerBlock(base_channels)
        self.temp_attn2 = TemporalTransformerBlock(base_channels * 2)
        self.temp_attn3 = TemporalTransformerBlock(base_channels * 4)
        self.temp_attn4 = TemporalTransformerBlock(base_channels * 8)
        self.temp_attn_bottleneck = TemporalTransformerBlock(base_channels * 16)

        # Convolution between temporal & spatial transformers
        self.conv_between1 = self.conv_block(base_channels, base_channels)
        self.conv_between2 = self.conv_block(base_channels * 2, base_channels * 2)
        self.conv_between3 = self.conv_block(base_channels * 4, base_channels * 4)
        self.conv_between4 = self.conv_block(base_channels * 8, base_channels * 8)
        self.conv_between_bottleneck = self.conv_block(base_channels * 16, base_channels * 16)

        # Spatial transformers
        self.spatial_attn1 = SpatialTransformerBlock(base_channels)
        self.spatial_attn2 = SpatialTransformerBlock(base_channels * 2)
        self.spatial_attn3 = SpatialTransformerBlock(base_channels * 4)
        self.spatial_attn4 = SpatialTransformerBlock(base_channels * 8)
        self.spatial_attn_bottleneck = SpatialTransformerBlock(base_channels * 16)


        # Encoder
        # Convolution applied per timestep
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
        self.temp_attention_blocks = nn.ModuleList([self.temp_attn1, self.temp_attn2, self.temp_attn3, self.temp_attn4])
        self.conv_between = nn.ModuleList([self.conv_between1, self.conv_between2, self.conv_between3, self.conv_between4])
        self.spatial_attention_blocks = nn.ModuleList([self.spatial_attn1, self.spatial_attn2, self.spatial_attn3, self.spatial_attn4])

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
    
    # Eventually apply 0/1/2 transformers depending on the attention strategy
    def apply_transformers(self, x_out, temp_bloc, conv, spatial_block):
        if "time" in self.attention:                     # Compute temporal attention 
            x_out = temp_bloc(x_out)        

            if "space" in self.attention:                  # If we perform both time & space attention, one should add a convolution block between those 2
                x_out = self.apply_conv_per_t(x_out, conv)

        if "space" in self.attention:                      # Compute spatial attention 
            x_out = spatial_block(x_out)

        return x_out


 
    def unet_forward(self, x): # Be careful, this is only the UNet block, the "real" forward is below
        # x: (B, T, C=2, H, W), we have two channels, the low res precip and the dem. T is the number of inputs

        # ===== Encoder =====
        encoder_outputs = []
        x_in = x
        for k in range(len(self.encoders)):
            x_out = self.apply_conv_per_t(x_in, self.encoders[k])
            encoder_outputs.append(x_out) # Save the convolution output to perform the future skip connection

            x_out = self.apply_transformers(x_out, self.temp_attention_blocks[k], self.conv_between[k], self.spatial_attention_blocks[k])   # Apply the temp/spatial transformers

            x_in = self.apply_pool_per_t(x_out)     # Max pooling

        # Bottleneck
        x_bottleneck = self.apply_conv_per_t(x_in, self.bottleneck)
        x_bottleneck = self.apply_transformers(x_bottleneck, self.temp_attn_bottleneck, self.conv_between_bottleneck, self.spatial_attn_bottleneck)

        # Keep only the low res frame to decode given that we don't compute (temporal) attention in the decoder
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

        return out     # (B, C, H, W) where C = temp_factor
    

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
            output_non_null = self.unet_forward(input) # output is [B', self.temp_factor, H, W]

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
                        output_final = f_output * (P_LR / sum_f)   # shape: (B', self.temp_factor, 100, 100)
                        output_non_null = output_final                      # Rename it

                # Setting the outputs
                outputs[non_null_mask.squeeze()] = output_non_null  # [B, self.temp_factor, H, W]


        else: # If all the batch is all zero (it might happened when the batch_size is low), we have to force the tensor to allow gradient computing
            outputs = torch.zeros(batch_size, self.temp_factor, 100, 100, device=frames[-1].device, requires_grad = True)  


        return outputs  









