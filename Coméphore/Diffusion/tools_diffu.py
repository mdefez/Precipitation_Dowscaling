# This file stores useful functions for the diffusion model

import torch
import torch.nn.functional as F


def setup_input(device, scheduler, A_seq, C, B, temporal_encoder): # Set up input/output for the diffusion model.
    A_seq = A_seq.to(device)            # (B, T, C_A, H, W) where C_A = 1. Output of the bicubic interpolation
    C = C.to(device)                    # (B, C_C, H, W) where C_C = temp_factor. Prediction of the deterministic model
    B = B.to(device)                    # (B, C_B, H, W) where C_B = temp_factor. Ground truth

    R = B - C                           # Residuals, (B, C, H, W)

    # échantillonner un timestep t pour chaque élément du batch
    B_size = B.size(0)
    t = torch.randint(0, scheduler.timesteps, (B_size,), device=device)     # Denoising step

    noise = torch.randn_like(R)
    R_t = scheduler.q_sample(R, t, noise=noise) # Noised residuals, (B, C, H, W)

    # encodeur temporel sur A pour extraire des features et coder l'aspect séquentiel
    temporal_input = A_seq.clone()  # (B, T, C, H, W)
    with torch.no_grad():
        temporal_embed = temporal_encoder(temporal_input)  # (B, T, D)

    # entrée du modèle : bruit R_t + dernière frames A_T et C (sortie du deterministic)
    A_T = A_seq[:, -1]                                # (B, C, H, W)
    model_input = torch.cat([R_t, A_T, C], dim=1)     # (B, C_B + C_A + C_C = 2*temp_factor + 2, H, W)

    # calcul de la vraie vélocité
    sqrt_alpha_bar = torch.sqrt(scheduler.alpha_bars[t]).view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - scheduler.alpha_bars[t]).view(-1, 1, 1, 1)

    velocity = sqrt_alpha_bar * noise - sqrt_one_minus_alpha_bar * R

    return model_input, temporal_embed, t, velocity

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

    B, T, C, H, W = list_frames[0].shape        # It's (B, 1, 1, H, W)

    # Change the shape to feed the interpolater
    frames_up = [frame.view(B * T, C, H, W) for frame in list_frames] 
    frames_up = [F.interpolate(frame, size=(100, 100), mode='bicubic', align_corners=False) for frame in frames_up] 
    frames_up = [frame.view(B, T, C, 100, 100) for frame in frames_up]

    frames_up = torch.cat(frames_up, dim = 1)

    return frames_up

# Apply conservative regridding between the prediction and the LR input
# The conservative regridding can be done either at the scale of the patch or the frame. 
# One must know applying regridding at the patch's scale might provide square artifacts
def apply_conservative_regridding_final_output(B_pred, LR_input, spatial_factor, temp_factor, hard_constraint_mass):

    strategy, f = hard_constraint_mass      # Scale (image or patch) and function f

    # This function apply conservative regridding for one block (temp_factor * spatial_factor * spatial_factor) VS low res pixel
    # x_block is (B, N, C*k*k) where C = temp_factor, y_pixel is (B, N, 1). We should perform the transformation for every B & N 
    # output should be the modified x_block (B, N, C*k*k)
    def apply_conservative_strategy_one_patch(x_block, y_pixel): 

        strategy, f = hard_constraint_mass

        f_output = f(x_block)  # shape: (B, N, C*k*k)

        # Compute the mass reference
        P_LR = y_pixel * temp_factor   # (B, N, 1)

        # Compute the sum at the denominator for the current predictions
        sum_f = f_output.sum(dim=(2), keepdim=True) / (spatial_factor ** 2)  #  (B, N, 1)

        # Compute the final (constrained) predictions
        output_final = f_output * (P_LR / sum_f)   # shape: (B, N, C*k*k)

        return output_final


    # Apply conservative reggriding LR pixel wise for the whole frame at the patch scale
    # Prediction is (B, C, H, W), LR last frame is (B, H', W') where C = temp_factor
    # Returns modified predictions of shape (B, C, H, W)
    def apply_conservative_strategy_patch_scale(prediction, last_frame):
        B, C, H, W = prediction.shape

        B, H_p, W_p = last_frame.shape

        # Get each temp_factor * (spatial_factor * spatial_factor) blocks 
        X_patches = F.unfold(prediction, kernel_size = spatial_factor, stride=spatial_factor)  # (B, C*k*k, N) where N = H'*W'
        X_patches = X_patches.transpose(1, 2)  # (B, N, C*k*k)

        # Resize the last frame for treatment
        Y_vals = last_frame.view(B, H_p * W_p, 1)        # (B, N, 1). Low res input 

        # Apply treatment by blocks
        X_patches_mod = apply_conservative_strategy_one_patch(X_patches, Y_vals)  # (B, N, C*k*k)

        # Resize to fit the expected shape
        X_patches_mod = X_patches_mod.transpose(1, 2)  # (B, C*k*k, N)
        X_out = F.fold(X_patches_mod, output_size=(H, W), kernel_size=spatial_factor, stride=spatial_factor) # (B, C, H, W)

        return X_out


    # Apply conservative reggriding at the frame scale
    # Prediction is (B, C, H, W), LR last frame is (B, H', W') where C = temp_factor
    # Returns modified predictions of shape (B, C, H, W)
    def apply_conservative_strategy_frame_scale(prediction, last_frame):
        strategy, f = hard_constraint_mass

        f_output = f(prediction)  # shape: (B, C, H, W)

        # Compute the mass reference
        P_LR = last_frame   # (B, H', W')
        P_LR = P_LR.sum(dim = (1, 2))       # (B)

        # Compute the sum at the denominator for the current predictions
        sum_f = f_output.sum(dim=(2, 3)) / (spatial_factor ** 2)  #  (B, C)

        # Make everything homogenous
        P_LR = P_LR.unsqueeze(1).unsqueeze(2).unsqueeze(3)        # (B, 1, 1, 1)
        sum_f = sum_f.unsqueeze(2).unsqueeze(3)        # (B, C, 1, 1)

        # Compute the final (constrained) predictions
        output_final = f_output * (P_LR / sum_f)   # shape: (B, C, H, W)

        return output_final
    
    LR_input = LR_input.squeeze()

    if strategy == "image-scale":
        final_output = apply_conservative_strategy_frame_scale(B_pred, LR_input)

    if strategy == "patch-scale":
        final_output = apply_conservative_strategy_patch_scale(B_pred, LR_input)

    return final_output
