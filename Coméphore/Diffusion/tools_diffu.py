# This file stores useful functions for the diffusion model

import torch
import torch.nn.functional as F


def setup_input(device, scheduler, A_seq, C, B, temporal_encoder): # Set up input/output for the diffusion model.
    A_seq = A_seq.to(device)            # (B, T, C_A, H, W) where C_A = 1, precip 
    C = C.to(device)                    # (B, C_C, H, W) where C_C = temp_factor
    B = B.to(device)                    # (B, C_B, H, W) where C_B = temp_factor

    R = B - C                           # résidu à apprendre, (B, C, H, W)

    # échantillonner un timestep t pour chaque élément du batch
    B_size = B.size(0)
    t = torch.randint(0, scheduler.timesteps, (B_size,), device=device)     # Denoising step

    noise = torch.randn_like(R)
    R_t = scheduler.q_sample(R, t, noise=noise)

    # encodeur temporel sur A pour extraire des features et coder l'aspect séquentiel
    temporal_input = A_seq.clone()  # (B, T, C, H, W)
    with torch.no_grad():
        temporal_embed = temporal_encoder(temporal_input)  # (B, T, D)

    # entrée du modèle : bruit R_t + dernière frames A_T et C (sortie du deterministic)
    A_T = A_seq[:, -1]                                # (B, C, H, W)
    model_input = torch.cat([R_t, A_T, C], dim=1)     # (B, C_B + C_A + C_C = 2*temp_factor + 2, H, W)

    return model_input, temporal_embed, t, noise

def setup_input_inference(device, R_t, A_seq, C, temporal_encoder): # Set up input/output for the diffusion model for the inference step
    A_seq = A_seq.to(device)            # (B, T, C_A, H, W) where C_A = 1, precip 
    C = C.to(device)                    # (B, C_C, H, W) where C_C = temp_factor
    R_t = R_t.to(device)                    # (B, C_B, H, W) where C_B = temp_factor


    # encodeur temporel sur A pour extraire des features et coder l'aspect séquentiel
    temporal_input = A_seq.clone()  # (B, T, C, H, W)
    with torch.no_grad():
        temporal_embed = temporal_encoder(temporal_input)  # (B, T, D)

    # entrée du modèle : bruit R_t + dernière frames A_T et C_T
    A_T = A_seq[:, -1]                                # (B, C, H, W)
    model_input = torch.cat([R_t, A_T, C], dim=1)     # (B, C_B + C_A + C_C = 2*temp_factor + 2, H, W)

    return model_input, temporal_embed


def bicubic_A_seq(list_frames): # Bicubically interpolate the list of frame to pass in the diffusion UNet
    B, T, C, H, W = list_frames[0].shape

    # Change the shape to feed the interpolater
    frames_up = [frame.view(B * T, C, H, W) for frame in list_frames] 
    frames_up = [F.interpolate(frame, size=(100, 100), mode='bicubic', align_corners=False) for frame in frames_up] 
    frames_up = [frame.view(B, T, C, 100, 100) for frame in frames_up]

    frames_up = torch.cat(frames_up, dim = 1)

    return frames_up
