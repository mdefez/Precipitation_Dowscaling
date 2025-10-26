import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Scheduler for the diffusion model. It's a classic one, one could choose between linear/quadratic increase for Betas
class DiffusionScheduler:
    def __init__(self, timesteps, type, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        if type == "linear":
            self.betas = torch.linspace(beta_start, beta_end, timesteps)
        if type == "quadratic":
            sqrt_beta_start = beta_start ** 0.5
            sqrt_beta_end = beta_end ** 0.5
            
            sqrt_betas = torch.linspace(sqrt_beta_start, sqrt_beta_end, timesteps)
            self.betas = sqrt_betas ** 2
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        # Send everything to the device
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        self.betas = self.betas.to(device)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        self.alpha_bars = self.alpha_bars.to(x_start.device)
        sqrt_ab = self.alpha_bars[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_ab = (1 - self.alpha_bars[t]).sqrt().view(-1, 1, 1, 1)

        return sqrt_ab * x_start + sqrt_one_minus_ab * noise

# This embeds each frame (C, H, W) of the sequence in a 256D "rich" vector
# Output is thus [B, T, 256] where T is the length of the sequence
# This is used in the encoder part to compute cross attention on
class TemporalEncoder(nn.Module):
    def __init__(self, input_channels, embed_dim, seq_len):
        super().__init__()
        self.cnn = nn.Conv2d(input_channels, embed_dim, kernel_size=3, padding=1)
        self.pos_embed = nn.Parameter(torch.randn(seq_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x_seq):  # x_seq: (B, T, C, H, W)
        B, T, C, H, W = x_seq.shape
        x_seq = x_seq.view(B * T, C, H, W)
        x = self.cnn(x_seq)  # (B*T, D, H, W), usual CNN
        x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)  # (B*T, D), we are making an average pooling on the image of dimensions (H, W)
        x = x.view(B, T, -1)  # (B, T, D)
        x = x + self.pos_embed.unsqueeze(0)
        x = self.transformer(x.transpose(0, 1))  # (T, B, D)
        return x.transpose(0, 1)  # (B, T, D)
    

# This functions is used to embed the denoising step t instead of using it as a scalar (the UNet can't use scalar as input easily, it must be embedded)
# It basically embedds the noising step into a vector of high dimension whose values are like trigonometric functions of t
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (B,) scalar t
        device = t.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)  # (half_dim,)
        emb = t[:, None].float() * emb[None, :]  # (B, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # (B, dim)

        return emb  # (B, dim), dim is the final dimension of the temporal vector


# This object allows to build a Positional Encoding Matrix to preserve the order within the tuple of tokens
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

    
# Usual way to compute temporal cross attention with multi attention heads 
# The feature map pays attention to the temporal embedding
class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, feature_dim, n_heads):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.positional_encoder = PositionalEncoding(embed_dim)

        self.query_proj = nn.Linear(feature_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, feature_dim)

    def forward(self, feature_map, temporal_embedding):  # (B, C, H, W), (B, T, D)
        B, C, H, W = feature_map.shape
        feat = feature_map.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)

        # We take into account the order sequence of the temporal embedding 
        temporal_embedding_encoded = self.positional_encoder(temporal_embedding)

        Q = self.query_proj(feat)  # (B, H*W, D)
        K = self.key_proj(temporal_embedding_encoded)  # (B, T, D)
        V = self.value_proj(temporal_embedding_encoded)  # (B, T, D)

        # Reshape for multi-head attention
        def reshape_for_heads(x):
            B, N, D = x.shape
            x = x.view(B, N, self.n_heads, self.head_dim)
            return x.permute(0, 2, 1, 3)  # (B, n_heads, N, head_dim)

        Q = reshape_for_heads(Q)
        K = reshape_for_heads(K)
        V = reshape_for_heads(V)

        attn_scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, n_heads, H*W, T)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = attn_weights @ V  # (B, n_heads, H*W, head_dim)

        # Concatenate heads
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # (B, H*W, n_heads, head_dim)
        attn_output = attn_output.view(B, -1, self.embed_dim)  # (B, H*W, D)

        out = self.out_proj(attn_output)  # (B, H*W, C)
        return (feat + out).permute(0, 2, 1).view(B, C, H, W)  # Residual connection



# Takes (B, T, C) as input and returns same size by passing it to a self attention mecanism
# B is the batch, T is the sequence of tokens, C are the token's channels.
# Each element of B is of channel C and will pay attention to the sequence of length T (where each element is also of channel C because it is self attention)
# This is the base architecture that will be used for spatial self attention below
class SelfAttentionBlock(nn.Module):
    def __init__(self, channels, num_heads, dropout=0.1, nb_features = 128):
        super().__init__()

        # Attention Matrix
        self.Q = nn.Linear(channels, nb_features)
        self.K = nn.Linear(channels, nb_features)
        self.V = nn.Linear(channels, nb_features)

        # Tensor X update
        self.norm1 = nn.LayerNorm(channels)  # Normalize over last dim (channels)
        self.attn = nn.MultiheadAttention(embed_dim=nb_features, num_heads=num_heads, batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(channels)

        self.ffn = nn.Sequential(
            nn.Linear(nb_features, channels * 4),  # (B, T, nb_features) → (B, T, 4C)
            nn.ReLU(),
            nn.Linear(channels * 4, channels),  # (B, T, 4C) → (B, T, C)
            nn.Dropout(dropout)
        )

    def forward(self, x):        
        # LayerNorm before attention
        x_norm = self.norm1(x)  # (B, T, C)

        # Compute (self) attention matrix
        query = self.Q(x_norm)      # (B, T, n_features)
        key = self.K(x_norm)        # (B, T, n_features)
        value = self.V(x_norm)      # (B, T, C)

        # Compute multi head attention output
        attn_output, _ = self.attn(query = query, key = key, value = value)  # (B, T, nb_features)

        attn_with_dropout = self.dropout(attn_output)

        # Feedforward network with residual
        ff_output = self.ffn(attn_with_dropout)  # (B, T, C)
        x = x + ff_output  # (B, T, C)

        return x  # Output shape: (batch_size, seq_len, channels)
    

# Takes (B, C, H, W) as input and returns same size by passing it to a spatial attention mecanism
# Every pixel of each frame will be transformed according to its neighbor values (from the same frame)
class LocalSpatialAttention(nn.Module):
    def __init__(self, channels, num_heads, window_size):
        super().__init__()

        self.window_size = window_size
        self.model = SelfAttentionBlock(channels = channels, num_heads = num_heads)

    def forward(self, x):
        B, C, H, W = x.shape

        if self.window_size == 1:       # No attention
            return x

        # Unfold to get local windows for each pixel
        patches = F.unfold(x, kernel_size = self.window_size, padding = self.window_size // 2)  # shape: (B, C*n², H*W) where n is the window_size

        patches = patches.transpose(1, 2).contiguous().view(B*H*W, self.window_size**2, C)  # (B*H*W, n², C)

        attn_output = self.model(patches)       # (B*H*W, n², C)

        # Take the center token (position in the patch)
        center_index = self.window_size**2 // 2
        output = attn_output[:, center_index, :].unsqueeze(1)  # (B*H*W, 1, C), we took the center of the window

        # Restore image shape
        output = output.transpose(1, 2).contiguous().view(B, C, H, W)    # (B, C, H, W)

        return output


# Weight standardized 2D convolution, prevents overfitting.
class WSConv2d(nn.Conv2d):
    def forward(self, x):
        # x: (B, C_in, H, W)
        weight = self.weight                         # (C_out, C_in, k, k)
        mean = weight.mean(dim=(1,2,3), keepdim=True)
        std = weight.std(dim=(1,2,3), keepdim=True) + 1e-5
        weight = (weight - mean) / std               # Weight standardized

        return nn.functional.conv2d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )  # Output: (B, C_out, H, W)

# Big convolutionnal block used in the diffusion UNet
# We use FiLM to take advantage of the denoising step t : We pass t into a MLP to extract 2 features gamma & beta that are used to scale the output of our convolutions
# Otherwise, we use 2 convs + ReLU
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = WSConv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = WSConv2d(out_channels, out_channels, 3, padding=1)

        self.gn = nn.GroupNorm(num_groups=8, num_channels=out_channels)     # (B, C_out, H, W)
        self.relu = nn.ReLU()

        # MLP pour gamma et beta à partir de l'embedding temps
        self.film = nn.Sequential(nn.Linear(time_emb_dim, 2 * out_channels),
                                  nn.ReLU(),
                                  nn.Linear(2 * out_channels, 2 * out_channels))

    def forward(self, x, t_emb):

        gamma_beta = self.film(t_emb)  # (B, 2*out_ch)
        gamma, beta = gamma_beta.chunk(2, dim=1)  # (B, out_ch), (B, out_ch)

        gamma = gamma[:, :, None, None]  # (B, out_ch, 1, 1)
        beta = beta[:, :, None, None]    # (B, out_ch, 1, 1)

        # FiLM : modulation canal par canal
        x1 = self.relu(self.gn(self.conv1(x)))
        x1 = gamma * x1 + beta  # (B, out_ch, H, W)

        x2 = self.relu(self.gn(self.conv2(x1)))
        out = gamma * (x2 + 1) + beta  # (B, out_ch, H, W), +1 is for residual approach

        return out, out.clone()  # return 2 to save a skip
    

# Global diffusion UNet
class UNetforDiffusion(nn.Module):
    def __init__(self, in_channels, base_channels, embed_dim, time_emb_dim, temp_factor, spatial_factor, window_size, nb_heads, strat_attention):
        super().__init__()
        # Attention factor
        self.strat_attention = strat_attention  # Attention strategy, wether to compute the temporal/spatial or not
        self.window_size = window_size          # Base size of the attention window (that can decrease over the encoder layers)
        self.nb_heads = nb_heads                # Number of heads for attention

        # SR factors
        self.temp_factor = temp_factor
        self.spatial_factor = spatial_factor

        # Denoising step as a vector
        self.time_emb = TimeEmbedding(time_emb_dim)

        # Encoder
        self.encoder1 = ConvBlock(in_channels, base_channels, time_emb_dim)
        self.encoder2 = ConvBlock(base_channels, base_channels * 2, time_emb_dim)
        self.encoder3 = ConvBlock(base_channels * 2, base_channels * 4, time_emb_dim)

        # Bottleneck
        self.encoder4 = ConvBlock(base_channels * 4, base_channels * 8, time_emb_dim)       

        # Pooling
        self.pool = nn.MaxPool2d(2)

        # Temporal cross attention
        self.temporal_attn_blocks = nn.ModuleList([CrossAttentionBlock(embed_dim, base_channels, n_heads=self.nb_heads), 
                                          CrossAttentionBlock(embed_dim, base_channels * 2, n_heads=self.nb_heads),            
                                          CrossAttentionBlock(embed_dim, base_channels * 4, n_heads=self.nb_heads), 
                                          CrossAttentionBlock(embed_dim, base_channels * 8, n_heads=self.nb_heads)])
        
        # Convolution between time & space attention
        self.conv_between_attention = nn.ModuleList([ConvBlock(base_channels, base_channels, time_emb_dim), 
                                                     ConvBlock(base_channels*2, base_channels*2, time_emb_dim),
                                                     ConvBlock(base_channels*4, base_channels*4, time_emb_dim),
                                                     ConvBlock(base_channels*8, base_channels*8, time_emb_dim)])

        # Spatial attention
        self.spatial_attn_blocks = nn.ModuleList([LocalSpatialAttention(channels = base_channels, num_heads = self.nb_heads, window_size = self.window_size[0]), 
                                                  LocalSpatialAttention(channels = base_channels*2, num_heads = self.nb_heads, window_size = self.window_size[1]),
                                                  LocalSpatialAttention(channels = base_channels*4, num_heads = self.nb_heads, window_size = self.window_size[2]),
                                                  LocalSpatialAttention(channels = base_channels*8, num_heads = self.nb_heads, window_size = self.window_size[3])])

        # Decoder. Each layer is the combination of a transposed convolution to upsample and a Convolutional block with FiLM
        self.upconv3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.decoder3 = ConvBlock(base_channels * 8, base_channels * 4, time_emb_dim)
        self.upconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.decoder2 = ConvBlock(base_channels * 4, base_channels * 2, time_emb_dim)
        self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.decoder1 = ConvBlock(base_channels * 2, base_channels, time_emb_dim)

        # Make them a list to simplify coding
        self.encoder = nn.ModuleList([self.encoder1, self.encoder2, self.encoder3, self.encoder4])
        self.decoder = nn.ModuleList([self.decoder1, self.decoder2, self.decoder3])
        self.upconv = nn.ModuleList([self.upconv1, self.upconv2, self.upconv3])

        # Final layer : 1*1 convolution
        self.final = nn.Sequential(nn.Conv2d(in_channels = base_channels, out_channels = self.temp_factor, kernel_size = 1))  # Predict temp_factor-channel residual (the noise)

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



    def forward(self, x, temporal_embedding, t):      # forward pass of the diffusion model

        t_emb = self.time_emb(t)  # (B, time_emb_dim), embedded denoising step

        # Encoder
        skips = []
        e = x
        for i in range(len(self.encoder)):
            e, skip = self.encoder[i](e, t_emb)                         # Remember the output for the skip connection

            if "time" in self.strat_attention:
                e = self.temporal_attn_blocks[i](e, temporal_embedding)     # Pay temporal cross attention to the whole (embedded) sequence

                if "space" in self.strat_attention:
                    e, _ = self.conv_between_attention[i](e, t_emb)                # Convolution 
            
            if "space" in self.strat_attention:
                e = self.spatial_attn_blocks[i](e)                          # Pay spatial self attention to your neighbors

            if i < len(self.encoder) - 1:
                skips.append(skip)
                e = self.pool(e)
            else:
                # Last encoder (= bottleneck) has no skip connection to keep
                skips.append(None)

        # Decoder
        d = e
        for i in reversed(range(len(self.decoder))):
            d = self.upconv[i](d)
            if skips[i] is not None:
                d = self.pad_to_match(d, skips[i])
                d = torch.cat([d, skips[i]], dim=1)
            d, _ = self.decoder[i](d, t_emb)

        output = self.final(d)

        return output
    


































