# The goal of this script is to implement a UNet class
# The model takes 3 channels as input (2 low res frames and DEM) and returns 6 high res frames

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

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
            #nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def up_conv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def last_layer(self, in_channels, interm_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, interm_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(interm_channels, out_channels, kernel_size=1), # 1x1 convolution as final step
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
        dec4 = torch.cat([dec4, conv4], dim=1)  # Skip connection

        dec3 = self.upconv3(dec4)
        dec3 = self.pad_to_match(dec3, conv3) 
        dec3 = torch.cat([dec3, conv3], dim=1)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, conv2], dim=1)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, conv1], dim=1)

        # Final layer
        output = self.output_layer(dec1)

        return output
    
    def forward(self, inp0, inp1, inp2): # This upsamples the 2 first channels and pass the 3 channels to the UNet

        # If both input are all zeroes, then we force the predictions to be zero. Otherwise, usual pipeline
        if (inp0.abs().sum() == 0) and (inp1.abs().sum() == 0):

            B = inp0.size(0)    
            device = inp0.device
            out_shape = (B, 3, 100, 100)  # We compute the output shape
            return torch.zeros(out_shape, device=device)

        # Upsample the 2 frames with bicubic interpolation
        inp0_up = F.interpolate(inp0, size=(100, 100), mode='bicubic', align_corners=False)
        inp1_up = F.interpolate(inp1, size=(100, 100), mode='bicubic', align_corners=False)

        # Concatenate the new input
        x_up = torch.cat([inp0_up, inp1_up, inp2], dim=1)

        # Pass it to the UNet
        return self.unet_forward(x_up)




