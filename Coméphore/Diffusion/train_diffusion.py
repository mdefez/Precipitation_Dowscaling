import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion_model import UNetforDiffusion, TemporalEncoder 
from forward import DiffusionScheduler
from dataset_diffusion import DummySequenceDataset

# Data needed for the diffusion model
# A = Sequence of interpolated frame (high res but bad).                    x = bicubic(low_res)          
# C = Output of the UNet, it is not a sequence because we only SR the last frame
# Actually, it can contain multiple frames if temp_factor > 1               \bar{y} = \mu(x) where \mu is the deterministic model
# B = Target = Last frame of the sequence                                   y 

# Features 
n_input = 3
temp_factor = 1



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train_diffusion_model(
    model, 
    temporal_encoder, 
    scheduler, 
    dataloader: DataLoader, 
    optimizer, 
    num_epochs, 
    device=device
):
    model.to(device)
    temporal_encoder.to(device)

    model.train()
    temporal_encoder.train()

    for epoch in range(num_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        epoch_loss = 0.0

        for A_seq, C, B in pbar:
            A_seq = A_seq.to(device)            # (B, T, C_A, H, W) where C_A = 2, precip and DEM
            C = C.to(device)                    # (B, C_C, H, W) where C_C = temp_factor
            B = B.to(device)                    # (B, C_B, H, W) where C_B = temp_factor

            R = B - C                           # résidu à apprendre, (B, C, H, W)

            # échantillonner un timestep t pour chaque élément du batch
            B_size = B.size(0)
            t = torch.randint(0, scheduler.timesteps, (B_size,), device=device)

            noise = torch.randn_like(R)
            R_t = scheduler.q_sample(R, t, noise=noise)

            # encodeur temporel sur A pour extraire des features et coder l'aspect séquentiel
            temporal_input = A_seq.clone()  # (B, T, C, H, W)
            with torch.no_grad():
                temporal_embed = temporal_encoder(temporal_input)  # (B, T, D)

            # entrée du modèle : bruit R_t + dernière frames A_T et C_T
            A_T = A_seq[:, -1]                                # (B, C, H, W)
            model_input = torch.cat([R_t, A_T, C], dim=1)     # (B, C_B + C_A + C_C = 2*temp_factor + 2, H, W)

            pred_noise = model(model_input, temporal_embed)
            loss = torch.nn.functional.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} — Avg Loss: {epoch_loss / len(dataloader):.6f}")

in_channels = 2*(temp_factor + 1)

model = UNetforDiffusion(in_channels=in_channels, base_channels=64, embed_dim=256)
temporal_encoder = TemporalEncoder(input_channels=2, embed_dim=256, seq_len=n_input) 
scheduler = DiffusionScheduler(timesteps=1000)

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(temporal_encoder.parameters()),
    lr=1e-4
)


dummy_dataset = DummySequenceDataset(
    num_samples=500, seq_len=5, channels=1, height=32, width=32
)

dataloader = DataLoader(dummy_dataset, batch_size=8, shuffle=True)


train_diffusion_model(model, temporal_encoder, scheduler, dataloader, optimizer, device = device, num_epochs=10)



























