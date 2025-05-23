import torch
import torch.nn as nn
from tqdm import tqdm

import sys

# Import functions from other files
sys.path.append('/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/Diffusion')
from tools_diffu import setup_input_inference, apply_conservative_regridding_final_output


@torch.no_grad()
def sample_diffusion(model, scheduler, A_seq, C, temporal_encoder, num_steps, last_frame, conservative_mass_diffusion, device="cuda"):
    B, T, Channel, H, W = A_seq.shape

    A_seq = A_seq.to(device)
    C = C.to(device) # (B, C, H, W) where C = temp_factor

    # Initial sample: R_T ~ N(0, I)
    R_t = torch.randn_like(C).to(device)
    

    for t_step in tqdm(
    reversed(range(num_steps)),
    desc="Sampling",
    total=num_steps,
    leave=True,
    ncols=100,
    file=sys.stdout):
        t = torch.full((B,), t_step, device=device, dtype=torch.long).to(device) # (B) tensor filled with t_step

        # Prédiction du bruit avec le modèle
        model_input, temporal_embed = setup_input_inference(device, R_t, A_seq, C, temporal_encoder)
        pred_epsilon = model(model_input, temporal_embed, t)  # (B, C, H, W)

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

        # Noise again to compute R_{t-1} (probabilistic approach)
        noise = torch.randn_like(R_t) if t_step > 0 else 0
        R_t = (
            torch.sqrt(alpha_bar_prev) * R_0_est +
            torch.sqrt(1 - alpha_bar_prev) * noise
        )

    # Given the residual, we can compute the true output
    B_pred = C + R_t
    B_pred = nn.ReLU()(B_pred)            # We can put ReLU because we want non negative output for precipitation

    # Conservative regridding
    B_pred = apply_conservative_regridding_final_output(B_pred = B_pred, LR_input = last_frame, spatial_factor = model.spatial_factor,
                                                        temp_factor = model.temp_factor, hard_constraint_mass = conservative_mass_diffusion)

    return B_pred
