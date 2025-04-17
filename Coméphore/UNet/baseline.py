# The goal of this script is to implement a UNet class
# The model takes 3 channels as input (2 low res frames and DEM) and returns 6 high res frames

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, hard_constraint_mass, temp_factor, spatial_factor):
        super(UNet, self).__init__()

        # Define the hard constraint mass strategy (None, additive or multiplicative)
        assert hard_constraint_mass == None or hard_constraint_mass == "additive" or len(hard_constraint_mass) == 2, "hard_constraint_mass has the wrong format"
        assert isinstance(temp_factor, int), "Wrong format for temporal SR factor"
        assert isinstance(spatial_factor, int), "Wrong format for spatial SR factor"
        
        self.hard_constraint_mass = hard_constraint_mass
        self.temp_factor = temp_factor
        self.spatial_factor = spatial_factor

        # Encoder
        self.encoder1 = self.conv_block(3, 64)  # 3 input channels 
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        self.encoder_pool = nn.MaxPool2d(2, 2)  # Max pooling layer, to put after each convolution

        # Bottleneck layer with no pooling, only convolution
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder 
        self.upconv4 = self.up_conv(1024, 512)
        self.upconv3 = self.up_conv(1024, 256) # The input is the concatenated (upconv above, encoder) so twice the output above
        self.upconv2 = self.up_conv(512, 128)
        self.upconv1 = self.up_conv(256, 64)

        # Output layer (Final layer for 6 predicted images)
        self.output_layer = self.last_layer(128, 64, 6) # We do two convolution in the last layer the variables are : input channels, intermediate channels, final channels


    def pad_to_match(self, tensor, target_tensor): # When decode, we eventually pad to 0 to match the corresponding encoder frame we concatenate on
        _, _, h, w = tensor.shape
        _, _, target_h, target_w = target_tensor.shape

        diff_y = target_h - h
        diff_x = target_w - w

        pad_left = diff_x // 2
        pad_right = diff_x - pad_left
        pad_top = diff_y // 2
        pad_bottom = diff_y - pad_top

        return F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom))


    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def up_conv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def last_layer(self, in_channels, interm_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, interm_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(interm_channels, out_channels, kernel_size=1), # 1x1 convolution as final step
            nn.ReLU(inplace=True) # Because we only seek for positive data
        )

    def unet_forward(self, x): # Be careful, this is only the UNet block, the "real" forward is below
        # Encoder 
        
        conv1 = self.encoder1(x)
        pool1 = self.encoder_pool(conv1) # We use two separate layers for convolution & pooling to keep track of the convolution's output, we can later concatenate in the decoder

        conv2 = self.encoder2(pool1)
        pool2 = self.encoder_pool(conv2)

        conv3 = self.encoder3(pool2)
        pool3 = self.encoder_pool(conv3)

        conv4 = self.encoder4(pool3)
        pool4 = self.encoder_pool(conv4)
        

        # Bottleneck 
        bottleneck = self.bottleneck(pool4)

        # Decoder 
        dec4 = self.upconv4(bottleneck)
        dec4 = self.pad_to_match(dec4, conv4)   # Pad with 0 so that we can concatenate the two tensors
        dec4 = torch.cat([dec4, conv4], dim=1)  # Skip connection

        dec3 = self.upconv3(dec4)
        dec3 = self.pad_to_match(dec3, conv3) 
        dec3 = torch.cat([dec3, conv3], dim=1)

        dec2 = self.upconv2(dec3)
        dec2 = self.pad_to_match(dec2, conv2) 
        dec2 = torch.cat([dec2, conv2], dim=1)

        dec1 = self.upconv1(dec2)
        dec1 = self.pad_to_match(dec1, conv1) 
        dec1 = torch.cat([dec1, conv1], dim=1)

        # Final layer
        output = self.output_layer(dec1)

        return output
    
    def forward(self, inp0, inp1, inp2): # This upsamples the 2 first channels and pass the 3 channels to the UNet

        ### FIRST STEP
        # Before anything, if both input images are all zeroes, then we force the predictions to be zero. Otherwise, we pass it to the unet

        # Create all outputs to zeros
        batch_size = inp0.size(0)
        outputs = torch.zeros(batch_size, 6, 100, 100, device=inp0.device)  
        
        # Check if for some sample, both input images are all zeros
        all_zero_inp0 = torch.all(inp0 == 0, dim=(-2, -1))  # [B]
        all_zero_inp1 = torch.all(inp1 == 0, dim=(-2, -1))  # [B]

        non_null_mask = ~(all_zero_inp0 & all_zero_inp1)  # [B], True if not entirely 0

        ### SECOND STEP
        # If not 0, pass them to the UNet. The UNet itself never sees any all zeroes frames
        if non_null_mask.any():
            # We only pass the non zero samples
            inp0_non_null = inp0[non_null_mask].unsqueeze(1)
            inp1_non_null = inp1[non_null_mask].unsqueeze(1)
            inp2_non_null = inp2[non_null_mask].unsqueeze(1)

            # Upsample the 2 frames with bicubic interpolation
            inp0_up = F.interpolate(inp0_non_null, size=(100, 100), mode='bicubic', align_corners=False)
            inp1_up = F.interpolate(inp1_non_null, size=(100, 100), mode='bicubic', align_corners=False)

            # Concat the non zeros sample to build a new "batch", with fewer samples than the inital ones (we made the all zeros go away)
            x_non_null = torch.cat((inp0_up, inp1_up, inp2_non_null), dim=1)  # [B', 3, H, W] où B' est le nombre d'échantillons non nuls

            # Pass it to the UNet

            output_non_null = self.unet_forward(x_non_null)

            ### THIRD STEP
            # Here we apply (if asked) the hard constraint mass strategy
            if self.hard_constraint_mass != None:
                if self.hard_constraint_mass == "additive":
                    pass 

                else: # This is thus the multiplicative strategy
                    strategy, f = self.hard_constraint_mass

                    f_output = f(output_non_null)  # shape: (B', 6, 100, 100)

                    # Be careful, here we take inp0 as the "initial Low Res" but it might be better to take inp1 or any mean, it depends on the strategy. 
                    P_LR = inp0_non_null.sum(dim=(2, 3), keepdim = True) * self.temp_factor     # Shape (B')

                    # Compute the sum at the denominator
                    sum_f = f_output.sum(dim=(1, 2, 3), keepdim=True) / (self.spatial_factor ** 2) # shape: (B', 1, 1, 1)

                    # Compute the final (constrained) outputs
                    output_final = f_output * (P_LR / sum_f)   # shape: (B', 6, 100, 100)
                    output_non_null = output_final                      # Rename it

            # Setting the outputs
            outputs[non_null_mask.squeeze()] = output_non_null  # [B, 6, H, W]


        else: # If all the batch is all zero (it might happened when the batch_size is low), we have to force the tensor to allow gradient computing
            outputs = torch.zeros(batch_size, 6, 100, 100, device=inp0.device, requires_grad = True)  

        return outputs  









