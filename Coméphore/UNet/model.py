import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder (Downsampling)
        self.encoder1 = self.conv_block(3, 64)  # 3 input channels (pour les 2 images basses résolutions + channel), 64 output channels
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        self.encoder_pool = nn.MaxPool2d(2, 2)  # Max pooling après chaque convolution

        # Bottleneck (Bottleneck layer with no pooling, only convolution)
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder (Upsampling)
        self.upconv4 = self.up_conv(1024, 512)
        self.upconv3 = self.up_conv(1024, 256) # The input is the concatenated (upconv above, encoder) so twice the output above
        self.upconv2 = self.up_conv(512, 128)
        self.upconv1 = self.up_conv(256, 64)

        # Output layer (Final layer for 6 predicted images)
        self.output_layer = self.last_layer(128, 64, 6) # input, inter & output channels


    def pad_to_match(self, tensor, target_tensor): # Gérer les arrondis de dimension lors du max pooling quand c'est impair
        _, _, h, w = tensor.shape
        _, _, target_h, target_w = target_tensor.shape

        diff_y = target_h - h
        diff_x = target_w - w

        # Padding = (left, right, top, bottom)
        pad_left = diff_x // 2
        pad_right = diff_x - pad_left
        pad_top = diff_y // 2
        pad_bottom = diff_y - pad_top

        return F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom))


    def conv_block(self, in_channels, out_channels):
        """Bloc de convolution pour le bottleneck (pas de MaxPooling ici)"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def up_conv(self, in_channels, out_channels):
        """Bloc de convolution transposé pour la décodification (upsampling)"""
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def last_layer(self, in_channels, interm_channels, out_channels):
        """Bloc de convolution pour le bottleneck (pas de MaxPooling ici)"""
        return nn.Sequential(
            nn.Conv2d(in_channels, interm_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(interm_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        # Encoder (Downsampling)
        conv1 = self.encoder1(x)
        pool1 = self.encoder_pool(conv1)

        conv2 = self.encoder2(pool1)
        pool2 = self.encoder_pool(conv2)

        conv3 = self.encoder3(pool2)
        pool3 = self.encoder_pool(conv3)

        conv4 = self.encoder4(pool3)
        pool4 = self.encoder_pool(conv4)

        # Bottleneck (sans MaxPooling, juste des convolutions)
        bottleneck = self.bottleneck(pool4)

        # Decoder (Upsampling)
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, conv4], dim=1)  # Concatenation avec enc4

        dec3 = self.upconv3(dec4)
        dec3 = self.pad_to_match(dec3, conv3) 
        dec3 = torch.cat([dec3, conv3], dim=1)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, conv2], dim=1)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, conv1], dim=1)

        # Sortie finale : 6 images prédites
        output = self.output_layer(dec1)

        return output




