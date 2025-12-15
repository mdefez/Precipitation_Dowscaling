# The goal of this script is to implement a UNet class with temporal/spatial attention

import torch
import torch.nn as nn
import torch.nn.functional as F


# General block to compute self-attention
# Takes (B, T, C) as input and returns same size by passing it to an attention mecanism
# B is the batch, T is the sequence of tokens, C are the token's channels
class SelfAttentionBlock(nn.Module):
    def __init__(self, channels, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)  # Normalize over last dim (channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(channels)

        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),  # (B, T, C) → (B, T, 4C)
            nn.ReLU(),
            nn.Linear(channels * 4, channels),  # (B, T, 4C) → (B, T, C)
            nn.Dropout(dropout)
        )

    def forward(self, x):        
        # LayerNorm before attention
        x_norm = self.norm1(x)  # (B, T, C)

        # Self-attention
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)  # (B, T, C)

        # Residual connection after attention
        x = x + self.dropout(attn_output)  # (B, T, C)

        # LayerNorm before feedforward
        x_norm = self.norm2(x)  # (B, T, C)

        # Feedforward network with residual
        ff_output = self.ffn(x_norm)  # (B, T, C)
        x = x + ff_output  # (B, T, C)

        return x  # Output shape: (batch_size, seq_len, channels)



# Compute temporal attention, it takes (B, T, C, H, W) as input and returns same size
# Every pixel of each frame will be transformed according to its previous values (from the other frames)
class TemporalAttention(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.model = SelfAttentionBlock(channels = channels, num_heads = num_heads)
        self.positional_encoding = PositionalEncoding(num_hiddens=channels)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.permute(0, 3, 4, 1, 2).contiguous()  # (B, H, W, T, C)
        x = x.view(B * H * W, T, C)  # (B*H*W, T, C), we compute attention for each B*H*W (pixel)

        x_positional_encoded = self.positional_encoding(x)      # Take the sequence order into account

        attn_output = self.model(x_positional_encoded)     # (B*H*W, T, C)
        good_shape = attn_output.view(B, H, W, T, C).permute(0, 3, 4, 1, 2).contiguous()  # (B, T, C, H, W)

        return good_shape




# Takes (B, T, C, H, W) as input and returns same size by passing it to a spatial attention mecanism
# Every pixel of the LAST FRAME will be transformed according to its neighbor values (from the same last frame)
class LocalSpatialAttention(nn.Module):
    def __init__(self, channels, num_heads, window_size):
        super().__init__()

        self.window_size = window_size
        self.model = SelfAttentionBlock(channels = channels, num_heads = num_heads)

    def forward(self, x):
        B, T, C, H, W = x.shape

        last_image = x.clone() 
        last_image = last_image[:, -1]  # Take last frame: (B, C, H, W)

        # Unfold to get local windows for each pixel
        patches = F.unfold(last_image, kernel_size = self.window_size, padding = self.window_size // 2)  # shape: (B, C*n², H*W)

        patches = patches.transpose(1, 2).contiguous().view(B*H*W, self.window_size**2, C)  # (B*H*W, n², C)

        attn_output = self.model(patches)       # (B*H*W, n², C)

        # Take the center token (position in the patch)
        center_index = self.window_size**2 // 2
        output = attn_output[:, center_index, :].unsqueeze(1)  # (B*H*W, 1, C), we took the center of the window

        # Restore image shape
        output = output.transpose(1, 2).contiguous().view(B, C, H, W)    # (B, C, H, W)

        # Returns the whole temporal sequence with the just modified last image
        x[:, -1] = output

        return x  # (B, T, C, H, W)


# This object allows to build a Positional Encoding Matrix to preserve the order of the sequence within the tuple of tokens
# We must specify max_len, the max number of tokens. We can set it to 15 given that it's the number of previous frames, usually set around 5.
class PositionalEncoding(nn.Module):  
    def __init__(self, num_hiddens, dropout=.1, max_len=15):      # num_hiddens is the dimension of the token's representation, in our case the number of channels
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Compute once and for all the encoding matrix P (We don't put it in the forward because it's the same, this way we compute it once)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        # Additive approach
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


# Deterministic UNet with temporal & spatial attention
class UNet_with_attention(nn.Module):
    def __init__(self, temp_factor, spatial_factor, model_parameters, input_channels=2, base_channels=16):
        super().__init__()
        
        self.temp_factor = temp_factor 
        self.spatial_factor = spatial_factor 
        self.hard_constraint_mass = model_parameters[0]
        self.n_inputs = model_parameters[1]
        self.attention = model_parameters[2]                # Attention strategy
        self.nb_heads = model_parameters[3]                 # Number of heads for the Multi head attention
        self.window_size = model_parameters[4]              # window size for spatial attention. Its a list where the i-th element is the size for the i-th layer

        self.base_channels = base_channels


        # Temporal Attention
        self.temp_attn1 = TemporalAttention(channels = base_channels, num_heads = self.nb_heads)
        self.temp_attn2 = TemporalAttention(channels = base_channels * 2, num_heads = self.nb_heads)
        self.temp_attn3 = TemporalAttention(channels = base_channels * 4, num_heads = self.nb_heads)
        self.temp_attn4 = TemporalAttention(channels = base_channels * 8, num_heads = self.nb_heads)
        self.temp_attn_bottleneck = TemporalAttention(channels = base_channels * 16, num_heads = self.nb_heads)

        # Convolution between temporal & spatial attention mecanism
        self.conv_between1 = self.conv_block(base_channels, base_channels)
        self.conv_between2 = self.conv_block(base_channels * 2, base_channels * 2)
        self.conv_between3 = self.conv_block(base_channels * 4, base_channels * 4)
        self.conv_between4 = self.conv_block(base_channels * 8, base_channels * 8)
        self.conv_between_bottleneck = self.conv_block(base_channels * 16, base_channels * 16)

        # Spatial attention
        self.spatial_attn1 = LocalSpatialAttention(channels = base_channels, num_heads = self.nb_heads, window_size = self.window_size[0])
        self.spatial_attn2 = LocalSpatialAttention(channels = base_channels * 2, num_heads = self.nb_heads, window_size = self.window_size[1])
        self.spatial_attn3 = LocalSpatialAttention(channels = base_channels * 4, num_heads = self.nb_heads, window_size = self.window_size[2])
        self.spatial_attn4 = LocalSpatialAttention(channels = base_channels * 8, num_heads = self.nb_heads, window_size = self.window_size[3])
        self.spatial_attn_bottleneck = LocalSpatialAttention(channels = base_channels * 16, num_heads = self.nb_heads, window_size = self.window_size[4])


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

        # Or we can use upsampling + conv 
        self.upsampling4 = self.upsample(base_channels * 16, base_channels * 8)
        self.upsampling3 = self.upsample(base_channels * 8, base_channels * 4)
        self.upsampling2 = self.upsample(base_channels * 4, base_channels * 2)
        self.upsampling1 = self.upsample(base_channels * 2, base_channels)

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

        # Choose the decoder strategy here
        self.up_decoder  = nn.ModuleList([self.upsampling4, self.upsampling3, self.upsampling2, self.upsampling1])
        self.decoders = nn.ModuleList([self.decoder4, self.decoder3, self.decoder2, self.decoder1])


        # Final layer, 1*1 convolution
        self.final_layer =  nn.Sequential(nn.Conv2d(self.base_channels, out_channels=self.temp_factor, kernel_size=1),
                                          nn.ReLU())

    # To match x's size to target before skip-connections
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
    

    def upsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

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
    
    # Eventually apply 0/1/2 attention mecanism depending on the strategy
    def apply_attention(self, x_out, temp_bloc, conv, spatial_block):
        if "time" in self.attention:                     # Compute temporal attention 
            x_out = temp_bloc(x_out)        

            if "space" in self.attention:                  # If we perform both time & space attention, one should add a convolution block between those 2
                x_out = self.apply_conv_per_t(x_out, conv)

        if "space" in self.attention:                      # Compute spatial attention 
            x_out = spatial_block(x_out)

        return x_out


 
    def unet_forward(self, x): # Be careful, this is only the UNet block, the "real" forward is below (especially, this unet_forward does not compute mass conservation)
        # x: (B, T, C=2, H, W), we have two channels, the low res precip and the dem. T is the number of inputs

        # Encoder
        encoder_outputs = []
        x_in = x
        for k in range(len(self.encoders)):
            x_out = self.apply_conv_per_t(x_in, self.encoders[k])
            encoder_outputs.append(x_out) # Save the convolution output to perform the future skip connection

            x_out = self.apply_attention(x_out, self.temp_attention_blocks[k], self.conv_between[k], self.spatial_attention_blocks[k])   # Apply the temp/spatial attention mecanism

            x_in = self.apply_pool_per_t(x_out)     # Max pooling

        # Bottleneck
        x_bottleneck = self.apply_conv_per_t(x_in, self.bottleneck)
        x_bottleneck = self.apply_attention(x_bottleneck, self.temp_attn_bottleneck, self.conv_between_bottleneck, self.spatial_attn_bottleneck)

        # Keep only the low res frame to decode given that we don't compute (temporal) attention in the decoder
        # This goes from (B, T, C, H, W) to (B, C, H, W)
        encoder_outputs = [f[:, -1] for f in encoder_outputs]
        x_bottleneck = x_bottleneck[:, -1]


        # Decoder
        x = x_bottleneck
        for i in range(len(self.up_decoder)):
            x = self.up_decoder[i](x)
            skip = encoder_outputs[-(i+1)]  
            x = self.pad_to_match(x, skip)
            x = torch.cat([x, skip], dim=1)
            x = self.decoders[i](x)

        # Final output
        out = self.final_layer(x)

        return out     # (B, C, H, W) where C = temp_factor
    


    # Apply conservative reggriding LR patch wise for the whole frame. This is not reccomended as it raises square artifacts
    # Prediction is (B, C, H, W), last frame is (B, 1, 1, H', W') where C = temp_factor
    # Returns modified predictions of shape (B, C, H, W)
    def apply_conservative_strategy_patch_scale(self, prediction, last_frame):
        B, C, H, W = prediction.shape

        _, _, _, H_p, W_p = last_frame.shape

        # Get each temp_factor * (spatial_factor * spatial_factor) blocks 
        X_patches = F.unfold(prediction, kernel_size=self.spatial_factor, stride=self.spatial_factor)  # (B, C*k*k, N) where N = H'*W'
        X_patches = X_patches.transpose(1, 2)  # (B, N, C*k*k)

        # Resize the last frame for treatment
        Y_vals = last_frame.view(B, 1, H_p * W_p).transpose(1, 2)  # (B, N, 1)

        # Apply treatment by blocks
        X_patches_mod = self.apply_conservative_strategy_one_patch(X_patches, Y_vals)  # (B, N, C*k*k)

        # Resize to fit the expected shape
        X_patches_mod = X_patches_mod.transpose(1, 2)  # (B, C*k*k, N)
        X_out = F.fold(X_patches_mod, output_size=(H, W), kernel_size=self.spatial_factor, stride=self.spatial_factor) # (B, C, H, W)

        return X_out

##############################################################################################################################
    # NB : Do not use patch-scale, it should be deleted entirely
##############################################################################################################################
 
    # This function apply conservative regridding for one patch (temp_factor * spatial_factor * spatial_factor) VS low res pixel
    # x_block is (B, N, C*k*k) where C = temp_factor, y_pixel is (B, N, 1). We should perform the transformation for every B & N 
    # output should be the modified x_block (B, N, C*k*k)
    def apply_conservative_strategy_one_patch(self, x_block, y_pixel): 

        strategy, f = self.hard_constraint_mass

        f_output = f(x_block)  # shape: (B, N, C*k*k)

        # If we only have low precip (below the specified treshold), we set f to identity
        if f_output.max() == 0:
            f_output = x_block

        # Compute the mass reference
        P_LR = y_pixel * self.temp_factor   # (B, N, 1)

        # Compute the sum at the denominator for the current predictions
        sum_f = f_output.sum(dim=(2), keepdim=True) / (self.spatial_factor ** 2)  #  (B, N, 1)

        # Compute the final (constrained) predictions
        output_final = f_output * (P_LR / sum_f)   # shape: (B, N, C*k*k)

        return output_final


    # Apply conservative reggriding LR pixel wise for the whole frame at the frame scale. This strategy is reccomended
    # Prediction is (B, C, H, W), last frame is (B, 1, 1, H', W') where C = temp_factor
    # Returns modified predictions of shape (B, C, H, W)
    def apply_conservative_strategy_frame_scale(self, prediction, last_frame):
        strategy, f = self.hard_constraint_mass

        f_output = f(prediction)  # shape: (B, C, H, W). At this point f_output can be null for a whole sample (C, H, W) which is not convenient for the following operations. In that case we simply don't apply the function

        # Compute the mass reference
        P_LR = last_frame.squeeze(2).squeeze(1)   # (B, H', W')
        P_LR = P_LR.sum(dim = (1, 2))       # (B)

        # Compute the sum at the denominator for the current predictions
        sum_f = f_output.sum(dim=(1, 2, 3))   #  (B)

        # Make everything homogenous
        P_LR = P_LR.unsqueeze(1).unsqueeze(2).unsqueeze(3)        # (B, 1, 1, 1)
        sum_f = sum_f.unsqueeze(1).unsqueeze(2).unsqueeze(3)        # (B, 1, 1, 1)

        mask = (sum_f > 1e-10).squeeze()      # One select only the B' non null batch

        P_lr_non_null = P_LR[mask]      
        f_output_non_null = f_output[mask]
        sum_non_null = sum_f[mask]

        # Compute the final (constrained) predictions
        output_final_non_null = f_output_non_null * (P_lr_non_null * self.temp_factor * (self.spatial_factor ** 2) / sum_non_null)   # shape: (B', C, H, W)
        output_final_null = prediction[~mask]

        output_final = torch.empty_like(prediction)

        output_final[mask] = output_final_non_null
        output_final[~mask] = output_final_null

        return output_final


    def forward(self, frames, dem, apply_constraint): # frames is a list of coarse inputs, dem is the dem associated to the tile

        strategy, f = self.hard_constraint_mass     # Mass conservation strategy

        ### First step
        # Before anything, if the last image (we seek to downsample) is all zeroes, then we force the predictions to be zero. Otherwise, we pass it to the unet

        # Create all outputs to zeros
        batch_size = frames[-1].size(0)

        outputs = torch.zeros(batch_size, self.temp_factor, 100, 100, device=frames[-1].device)  
        
        # Check if for some sample, the two last input is all zeros
        all_zero_inp0 = torch.all(frames[-1] == 0, dim=(-2, -1))  # [B]

        non_null_mask = ~(all_zero_inp0).squeeze()  # [B], True if not entirely 0

        ### Second step
        # If not 0, pass them to the UNet. The UNet itself never sees any all zeroes frames
        if non_null_mask.any():
            # We only pass the non zero samples
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

            ### Third step
            # Here we apply (if asked) the hard constraint mass strategy
            if apply_constraint == True:
                if strategy == "patch-scale":
                    output_non_null = self.apply_conservative_strategy_patch_scale(prediction=output_non_null,
                                                            last_frame=frames_non_null[-1])       # [B', self.temp_factor, H, W]
                    
                if strategy == "image-scale":
                    output_non_null = self.apply_conservative_strategy_frame_scale(prediction=output_non_null,
                                                            last_frame=frames_non_null[-1])       # [B', self.temp_factor, H, W]

            # Setting the outputs
            outputs[non_null_mask.squeeze()] = output_non_null  # [B, self.temp_factor, H, W]


        else: # If all the batch is all zero (it might happened when the batch_size is low), we have to force the tensor to allow gradient computing
            outputs = torch.zeros(batch_size, self.temp_factor, 100, 100, device=frames[-1].device, requires_grad = True)  


        return outputs  









