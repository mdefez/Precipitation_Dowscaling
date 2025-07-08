# This scripts aims to define a custom loss function implementing some options

import torch 
import torch.nn as nn
import numpy as np
import os
from scipy.stats import wasserstein_distance
import copy
import matplotlib.pyplot as plt
from torchmetrics.functional import structural_similarity_index_measure as ssim_f

import warnings
warnings.filterwarnings(
    "ignore",
    message="Importing `spectral_angle_mapper` from `torchmetrics.functional` was deprecated"
)

class CustomLossTrain(nn.Module):
    def __init__(self, base_loss, conservative = False, lambda_conservative = 0.1, lambda_covariance = 0.1, covariance = False):
        super().__init__()

        self.loss_mean = copy.deepcopy(base_loss)
        self.loss_none = copy.deepcopy(base_loss)

        # On force les modes de réduction
        self.loss_mean.reduction = 'mean'
        self.loss_none.reduction = 'none'

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
        loss = self.loss_mean(outputs, targets) 

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
    

class LossTest(nn.Module):          # We us this loss function as a metric on the training set. It allows to compute averaged and marginal loss over the batch
    def __init__(self, df_metric):
        super().__init__()
        self.name_metric = list(df_metric["Name"])
        self.metric = [metric() for metric in df_metric["Metric"]]

        # Use the main metric to compute best/worst examples
        self.loss_vector = copy.deepcopy(self.metric[0])

        # We specifiy reduction = None to compute the loss as a vector
        self.loss_vector.reduction = 'none'

    def forward(self, outputs, targets): # Usual (averaged) loss for each metric
        loss = np.array([metric(outputs, targets).item() for metric in self.metric])

        return loss


    def forward_vecteur(self, outputs, targets): # Compute the loss as a vector (only for the main metric), useful for selecting the "best" and "worst" samples
        loss = self.loss_vector(outputs, targets).mean(dim = (1, 2, 3)) 

        return loss 
    
    def crps(self, list_outputs, target, v_func = lambda x: x**2):      # Compute the CRPS over a list of outputs (list of length the number of scenarios, each item is a batch)

        # Transform the list of outputs as a tensor [M, B]
        ensemble_tensor = torch.stack(list_outputs)  # [M, B]
        M = ensemble_tensor.size(0)

        # Apply the function v
        v_ensemble = v_func(ensemble_tensor)     # [M, B]
        v_y = v_func(target).unsqueeze(0)             # [1, B] 

        # First quantity : Bias
        term1 = torch.mean(torch.abs(v_ensemble - v_y), dim=0)  # [B]

        # Second quantity : Spread between scenarios
        diff = v_ensemble.unsqueeze(1) - v_ensemble.unsqueeze(0)  # [M, M, B]
        term2 = torch.mean(torch.abs(diff), dim=(0,1))  # [B]

        return (term1 - 0.5 * term2).mean()         # mean over the batch

    

# Computes the absolute difference between the 99th percentiles of output and target for each image/channel, and averages channels (and eventually batch depending on the reduction)
class PercentileDifferenceLoss(nn.Module):
    def __init__(self, percentile = 99):
        super().__init__()

        self.percentile = percentile

    def forward(self, output, target):

        B, C, H, W = output.shape

        output_flat = output.view(B, C, -1)     # [B, C, H*W]
        target_flat = target.view(B, C, -1)   # [B, C, H*W]

        output_p = torch.quantile(output_flat, self.percentile / 100.0, dim=2)   # [B, C]
        target_p = torch.quantile(target_flat, self.percentile / 100.0, dim=2) # [B, C]

        diff = torch.abs(output_p - target_p)  # [B, C]
        loss_per_image = diff.mean()     # scalar, mean over channels & batch


        return loss_per_image.mean()      # scalar




    

# Compute the LSD between pred & target
class Log_spectral_distance(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def precompute_bin_indices_torch(self, h, w, bins=128):
        y, x = torch.meshgrid(torch.arange(h, device=self.device), torch.arange(w, device=self.device), indexing='ij')
        center = (h // 2, w // 2)
        r = torch.sqrt((x - center[1])**2 + (y - center[0])**2)
        bin_edges = torch.linspace(0, r.max(), bins + 1, device=self.device)
        bin_indices = torch.bucketize(r.flatten(), bin_edges) - 1  # (H*W,)
        return bin_indices, bins

    # From 2D FFT to 1D FFT by averaging radially
    def batch_radial_average(self, images, bin_indices, bins):

        B, C, H, W = images.shape
        images_flat = images.reshape(B, C, -1)  # (B, C, H*W)

        radial_means = torch.zeros((B, C, bins), device=images.device, dtype=images.dtype)

        for b in range(bins):
            mask = (bin_indices == b)
            # Sum pixels in the bin for each B and C
            sums = images_flat[:, :, mask].sum(dim=2)  # (B, C)
            counts = mask.sum().item()
            radial_means[:, :, b] = sums / (counts + 1e-16)

        return radial_means

    # Compute LSD over the whole batch
    def forward(self, predictions, targets, bins=128, epsilon=1e-8):     # output and target are (B, C, H, W)

        B, C, H, W = predictions.shape

        # Compute FFT magnitude
        fft_pred = torch.fft.fftshift(torch.fft.fft2(predictions, dim=(-2, -1)), dim=(-2, -1))      #  (B, C, H, W), complex
        fft_target = torch.fft.fftshift(torch.fft.fft2(targets, dim=(-2, -1)), dim=(-2, -1))

        mag_pred = torch.abs(fft_pred)      #  (B, C, H, W), float
        mag_target = torch.abs(fft_target)

        # Precompute bin indices for radius bins (same for all images)
        bin_indices, bins = self.precompute_bin_indices_torch(H, W, bins)

        # Radial averaging
        radial_pred = self.batch_radial_average(mag_pred, bin_indices, bins)            
        radial_target = self.batch_radial_average(mag_target, bin_indices, bins)

        # Log of radial spectrum (epsilon for numerical stability)
        log_radial_pred = torch.log(radial_pred + epsilon)          # (B, C, bins)
        log_radial_target = torch.log(radial_target + epsilon)

        # Compute the mean LSD over (B, C)
        lsd_per_channel_image = torch.sqrt(torch.mean((log_radial_pred - log_radial_target) ** 2, dim=2))       # (B, C)
        mean_lsd = lsd_per_channel_image.mean()  # scalar, mean over batch & channels

        return mean_lsd # scalar



# Compute the EMD after normalizing frames
class EarthMovingDistance(nn.Module):
    def __init__(self):
        super().__init__()

    # To compute proper EMD, the frame must sum up to 1
    def normalize_images(self, imgs, epsilon=1e-8):

        imgs_flat = imgs.flatten(start_dim=2)  # (B, C, H*W)
        sums = imgs_flat.sum(dim=2, keepdim=True) + epsilon  # (B, C, 1)
        return imgs_flat / sums  # (B, C, H*W)

    def forward(self, predictions, targets):        # output and target are (B, C, H, W)

        B, C, H, W = predictions.shape
        device = predictions.device 

        pred_norm = self.normalize_images(predictions)  # (B, C, H*W)
        target_norm = self.normalize_images(targets)  # (B, C, H*W)

        y, x = torch.meshgrid(torch.arange(H, device="cpu"), torch.arange(W, device="cpu"), indexing='ij')        # y, x: (H, W)
        
        # We need the coords to identify the "ground" and the associated distances between pixel
        coords = torch.stack([x.flatten(), y.flatten()], dim=1).float()  # (H*W, 2)

        emd_vals = torch.zeros((B, C), device="cpu")  # (B, C)

        # Loop because the function doesn't handle tensors
        for b in range(B):
            for c in range(C):
                p_dist = pred_norm[b, c].to("cpu")  # (H*W,)
                q_dist = target_norm[b, c].to("cpu")  # (H*W,)
                emd_vals[b, c] = wasserstein_distance(p_dist, q_dist, coords, coords)  # scalar

        mean_emd = emd_vals.mean()  # scalar, mean over batch & channels

        return mean_emd     # scalar



# Compute SSIM (to evaluate the "realism" of the image) depending on the structure
# This metric should be computed for each frame (H, W) so we iterate over the batch and the channels then average over it   
# Between 0 and 1 (theoritically it could go to -1 for anticorrelation). 1 is the best 
class SSIM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):    # output and target are (B, C, H, W)

        B, C, H, W = pred.shape

        # Flatten time dimension: (B*C, 1, H, W)
        pred_flat = pred.reshape(B * C, 1, H, W)
        target_flat = target.reshape(B * C, 1, H, W)

        # Compute SSIM for each grayscale image
        ssim_vals = ssim_f(
            pred_flat,
            target_flat,
            data_range=1,
            gaussian_kernel=True,
            sigma=1.5,
            kernel_size=11,
            reduction="none",  # returns tensor of shape (B*C,)
        )

        return ssim_vals.mean()
    


# Compute the rank histogram and the associated deviation 
# We compute a rank histogram for each image (channel). One can eventually plot it 
class PITD(nn.Module):
    def __init__(self):
        super().__init__()

    # Compute (and eventually plot) the rank histogram for one image. The "real" forward is below.
    def compute_rank_histogram(self, output, target, bins=None, plot_histo = False, plot_path = None):      # output and target are (H, W)

        # Flatten the arrays
        pred_values = np.ravel(output.cpu().detach().numpy())
        target_values = np.ravel(target.cpu().detach().numpy())

        # Sort prediction values once
        pred_sorted = np.sort(pred_values)
        
        # Compute ranks
        ranks = np.searchsorted(pred_sorted, target_values, side="right")

        # By default, bins = N_pred + 1
        if bins is None:
            bins = len(pred_sorted) + 1

        # Histogram counts
        counts, bin_edges = np.histogram(ranks, bins=bins)
        expected_freq = 1 / bins

        # Plot histogram
        if plot_histo:
            plt.figure(figsize=(8, 5))
            plt.bar(
                range(bins),
                counts / counts.sum(),
                width=1,
                edgecolor="black",
                align="center"
            )
            plt.axhline(y=expected_freq, color="red", linestyle="--", linewidth=2, label="Expected Uniform Frequency")
            plt.xlabel("Rank Bin")
            plt.ylabel("Relative Frequency")
            plt.title("Rank Histogram")
            plt.savefig(plot_path)

        # Computes the deviation
        K = bins
        N = counts.sum()
        expected = 1 / K
        pitd = np.sqrt(np.mean((counts / N - expected) ** 2))

        return pitd
    
    # Plot every channel's histograms for one sample of the batch
    # The sample is choosen so that it is non null and contains significant amount of precipitation
    def plot_channels(self, output, target, plot_path, bins = 10):     # Output & target are (B, C, H, W)
        B, C, H, W = output.shape
        os.makedirs(plot_path, exist_ok=True)

        for b in range(B):
            local_frame = output[b, :, :, :]

            if local_frame.mean() > 0.1:        # Significant amount of precipitation
                for c in range(C):
                    local_output = output[b, c, :, :]
                    local_target = target[b, c, :, :]
                    local_plot_path = plot_path + f"timestep_{c}.png"
                    self.compute_rank_histogram(output = local_output,
                                                            target = local_target,
                                                            plot_histo = True,
                                                            bins=bins,
                                                            plot_path=local_plot_path)
                    
                break
                        
                

    
    # Compute the PITD for each batch and channel then average it 
    def forward(self, pred, target):    # output and target are (B, C, H, W)

        B, C, H, W = pred.shape

        mean_pitd = 0
        for b in range(B):
            for c in range(C):
                output = pred[b, c, :, :]
                local_target = target[b, c, :, :]
                mean_pitd += self.compute_rank_histogram(output = output,
                                                        target = local_target,
                                                        plot_histo = False)
                
        mean_pitd = mean_pitd / (B*C)

        return mean_pitd














