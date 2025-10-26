import torch
import torch.nn as nn
from tqdm import tqdm

import sys

# Import functions from other files
from Coméphore.Diffusion.tools_diffu import setup_input_inference, apply_conservative_regridding_final_output


@torch.no_grad()
def sample_diffusion(model, scheduler, A_seq, C, temporal_encoder, num_steps, last_frame, conservative_mass_diffusion, n_scenarios, device):
    B, T, Channel, H, W = A_seq.shape

    # Store the n possible scenarios in a list
    list_scenarios = []

    A_seq = A_seq.to(device)
    C = C.to(device) # (B, C, H, W) where C = temp_factor

    # Compute one scenario
    for _ in range(n_scenarios):
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
            pred_velo = model(model_input, temporal_embed, t)  # (B, C, H, W)

            alpha_t = scheduler.alphas[t_step].view(-1, 1, 1, 1)
            alpha_bar_t = scheduler.alpha_bars[t_step].view(-1, 1, 1, 1)
            beta_t = scheduler.betas[t_step].view(-1, 1, 1, 1)

            # Compute epsilon directly from predicted velocity and R_t
            epsilon_pred = torch.sqrt(alpha_bar_t) * pred_velo + torch.sqrt(1 - alpha_bar_t) * R_t
        

            # Calculate posterior mean mu_t
            coef1 = 1 / torch.sqrt(alpha_t)
            coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
            mu_t = coef1 * (R_t - coef2 * epsilon_pred)
            
            if t_step > 1:   # Noise again to keep the probabilistic approach
                sigma_t = torch.sqrt(beta_t)
                noise = torch.randn_like(R_t)
                R_t = mu_t + sigma_t * noise
            else:       # Don't noise because it's the final step
                R_t = mu_t


        # Given the residual, we can compute the true output
        B_pred = C + R_t
        B_pred = nn.ReLU()(B_pred)            # We can put ReLU because we want non negative output for precipitation

        # Conservative regridding
        B_pred = apply_conservative_regridding_final_output(B_pred = B_pred, LR_input = last_frame, spatial_factor = model.spatial_factor,
                                                            temp_factor = model.temp_factor, hard_constraint_mass = conservative_mass_diffusion)

        list_scenarios.append(B_pred)

    return list_scenarios
