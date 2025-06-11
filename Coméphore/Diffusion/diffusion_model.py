import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

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

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        self.alpha_bars = self.alpha_bars.to(x_start.device)
        sqrt_ab = self.alpha_bars[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_ab = (1 - self.alpha_bars[t]).sqrt().view(-1, 1, 1, 1)

        return sqrt_ab * x_start + sqrt_one_minus_ab * noise


# Pour coder chaque image de la séquence (C, H, W) en un vecteur riche de taille 256. Ca sort donc un vecteur [B, T, 256].
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
        x = self.cnn(x_seq)  # (B*T, D, H, W), CNN classique
        x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)  # (B*T, D), on fait un average pooling sur tout (H, W)
        x = x.view(B, T, -1)  # (B, T, D)
        x = x + self.pos_embed.unsqueeze(0)
        x = self.transformer(x.transpose(0, 1))  # (T, B, D)
        return x.transpose(0, 1)  # (B, T, D)
    
# --- Embedding sinusoidal du temps t --- pour que le UNet prenne en compte le step de diffusion
# On code le step t en un vecteur de grande dimension dont les composantes sont des fonctions du step t (plus ou moins des fonctions trigos)
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (B,) entier scalaire temps (step)
        device = t.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)  # (half_dim,)
        emb = t[:, None].float() * emb[None, :]  # (B, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # (B, dim)

        return emb  # (B, dim)


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, feature_dim):
        super().__init__()
        self.query_proj = nn.Linear(feature_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, feature_dim)

    def forward(self, feature_map, temporal_embedding):  # (B, C, H, W), (B, T, D)
        B, C, H, W = feature_map.shape
        feat = feature_map.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        queries = self.query_proj(feat)  # (B, H*W, D)
        keys = self.key_proj(temporal_embedding)  # (B, T, D)
        values = self.value_proj(temporal_embedding)  # (B, T, D)

        attn = torch.softmax(queries @ keys.transpose(-2, -1) / (C ** 0.5), dim=-1)
        attended = attn @ values  # (B, H*W, D)
        out = self.out_proj(attended)  # (B, H*W, C)
        return (feat + out).permute(0, 2, 1).view(B, C, H, W)  # Residual


# Weight standardized
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


class UNetforDiffusion(nn.Module):
    def __init__(self, in_channels, base_channels, embed_dim, time_emb_dim, temp_factor, spatial_factor):
        super().__init__()
        self.temp_factor = temp_factor
        self.spatial_factor = spatial_factor

        # Denoising step as a vector
        self.time_emb = TimeEmbedding(time_emb_dim)

        # Encoder
        self.encoder1 = ConvBlock(in_channels, base_channels, time_emb_dim)
        self.encoder2 = ConvBlock(base_channels, base_channels * 2, time_emb_dim)
        self.encoder3 = ConvBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        self.encoder4 = ConvBlock(base_channels * 4, base_channels * 8, time_emb_dim)       # Bottleneck

        self.encoder = nn.ModuleList([self.encoder1, self.encoder2, self.encoder3, self.encoder4])

        # Pooling
        self.pool = nn.MaxPool2d(2)

        # Temporal attention
        self.attn_blocks = nn.ModuleList([CrossAttentionBlock(embed_dim, base_channels), CrossAttentionBlock(embed_dim, base_channels * 2),
            CrossAttentionBlock(embed_dim, base_channels * 4), CrossAttentionBlock(embed_dim, base_channels * 8)])

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.decoder3 = ConvBlock(base_channels * 8, base_channels * 4, time_emb_dim)
        self.upconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.decoder2 = ConvBlock(base_channels * 4, base_channels * 2, time_emb_dim)
        self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.decoder1 = ConvBlock(base_channels * 2, base_channels, time_emb_dim)

        self.decoder = nn.ModuleList([self.decoder1, self.decoder2, self.decoder3])
        self.upconv = nn.ModuleList([self.upconv1, self.upconv2, self.upconv3])

        # Final layer
        self.final = nn.Sequential(nn.Conv2d(in_channels = base_channels, out_channels = self.temp_factor, kernel_size = 1))  # Predict 1-channel residual (the noise)

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



    def forward(self, x, temporal_embedding, t):      # "Real" diffusion model, the real forward is right after

        t_emb = self.time_emb(t)  # (B, time_emb_dim), denoising step

        # --- Encoder ---
        skips = []
        e = x
        for i in range(len(self.encoder)):
            e, skip = self.encoder[i](e, t_emb)
            e = self.attn_blocks[i](e, temporal_embedding)  # Pay attention to the whole (embedded) sequence
            if i < len(self.encoder) - 1:
                skips.append(skip)
                e = self.pool(e)
            else:
                # Last encoder (= bottleneck) has no skip connection to keep
                skips.append(None)

        # --- Decoder ---
        d = e
        for i in reversed(range(len(self.decoder))):
            d = self.upconv[i](d)
            if skips[i] is not None:
                d = self.pad_to_match(d, skips[i])
                d = torch.cat([d, skips[i]], dim=1)
            d, _ = self.decoder[i](d, t_emb)

        output = self.final(d)

        return output
    


































