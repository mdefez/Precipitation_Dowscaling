# This scripts aims to define a custom loss function implementing some options

import torch 
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, base_loss, conservative = False, lambda_conservative = 0.1, lambda_covariance = 0.1, covariance = False):
        super().__init__()
        self.base_loss = base_loss

        self.lambda_conservative = lambda_conservative
        self.lambda_covariance = lambda_covariance
        self.conservative = conservative
        self.covariance = covariance


    # Returns the temporal autocorrelation of the 6 channels
    # Output is (B, H, W, 6, 6) because there is a 6*6 matrix for each pixel (storing the autocorrelation factors for every lags)
    def compute_autocorr_matrix(self, x):

        x = x.permute(0, 2, 3, 1)  # [B, H, W, T]
        x = x.unsqueeze(-1)        # [B, H, W, T, 1]
        x_T = x.transpose(-2, -1)  # [B, H, W, 1, T]

        # Outer product: [B, H, W, T, T]
        autocorr = x @ x_T

        return autocorr  # [B, H, W, 6, 6]

    def forward(self, outputs, targets):
        loss = self.base_loss(outputs, targets)

        if self.conservative == True:
            # Conservative term
            sum_outputs = outputs.sum(dim=(1, 2, 3))  # Sum over the channels + whole frames
            sum_targets = targets.sum(dim=(1, 2, 3))  # Same for target

            conservative_term = torch.abs(sum_outputs - sum_targets).mean()  # Compute the difference and average over the whole batch

            loss += self.lambda_conservative * conservative_term

        if self.covariance == True:
            # Autocorrelative term
            ac_out = self.compute_autocorr_matrix(outputs)  # [B, H, W, 6, 6]
            ac_tar = self.compute_autocorr_matrix(targets)  # [B, H, W, 6, 6]

            # Frobenius norm of difference
            frob_diff = torch.norm(ac_out - ac_tar, dim=(-2, -1))  # [B, H, W]
            frob_loss = frob_diff.mean()  # Mean over the batch & pixels

            loss += self.lambda_covariance * frob_loss

        return loss 
