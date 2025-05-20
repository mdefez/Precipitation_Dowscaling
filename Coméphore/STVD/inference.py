import torch
from tqdm import tqdm

import sys

# Import functions from other files
sys.path.append('/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/Diffusion')

from diffusion_model import TemporalEncoder, setup_input_inference

@torch.no_grad()
def sample_diffusion(model, scheduler, A_seq, C, n_input, num_steps=1000, device="cuda"):
    model.eval()
    B, T, Channel, H, W = A_seq.shape

    A_seq = A_seq.to(device)
    C = C.to(device) # (B, C, H, W)

    # Initial sample: R_T ~ N(0, I)
    R_t = torch.randn_like(C).to(device)
    temporal_encoder = TemporalEncoder(input_channels=1, embed_dim=256, seq_len=n_input).to(device).train()

    for t_step in tqdm(
    reversed(range(num_steps)),
    desc="Sampling",
    total=num_steps,
    leave=True,
    ncols=100,
    file=sys.stdout):
        t = torch.full((B,), t_step, device=device, dtype=torch.long).to(device)

        # Prédiction du bruit avec le modèle
        model_input, temporal_embed = setup_input_inference(device, R_t, A_seq, C, temporal_encoder)
        pred_epsilon = model(model_input, temporal_embed)  # (B, C, H, W)

        # Scheduler params
        alpha_bars = scheduler.alpha_bars.to(device)
        alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1).to(device)
        alpha_bar_prev = alpha_bars[torch.clamp(t-1, 0)].view(-1, 1, 1, 1).to(device)

        beta_t = 1 - alpha_bar_t / alpha_bar_prev
        beta_t = torch.clamp(beta_t, 0.0001, 0.9999)  # sécurité numérique

        # Posterior mean estimate for R_{t-1}
        coef1 = (1 / torch.sqrt(alpha_bar_t))
        coef2 = (1 - alpha_bar_t).sqrt()
        R_0_est = (R_t - coef2 * pred_epsilon) / coef1

        # Rebruiter pour obtenir R_{t-1}
        noise = torch.randn_like(R_t) if t_step > 0 else 0
        R_t = (
            torch.sqrt(alpha_bar_prev) * R_0_est +
            torch.sqrt(1 - alpha_bar_prev) * noise
        )

    # Une fois R_0 obtenu → reconstruire B_T
    B_pred = C + R_t

    return B_pred
