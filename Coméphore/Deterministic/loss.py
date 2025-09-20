# This scripts aims to define custom loss function that are used to evaluate models

import torch 
import torch.nn as nn
import numpy as np
import os
from scipy.stats import wasserstein_distance
import copy
import matplotlib.pyplot as plt
from torchmetrics.functional import structural_similarity_index_measure as ssim_f
import pandas as pd

import warnings
warnings.filterwarnings(
    "ignore",
    message="Importing `spectral_angle_mapper` from `torchmetrics.functional` was deprecated"
)

    
# We use this loss function as a metric on the training set. It allows to compute averaged or marginal loss over the batch
class LossTest(nn.Module):          
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
        loss = self.loss_vector(outputs, targets).mean(dim = (1, 2, 3))         # (B)

        return loss 
    
    # Compute the CRPS over a list of outputs (list of length the number of scenarios, each item is a batch)
    def crps(self, list_outputs, target, v_func = lambda x: x**2):      

        # Transform the list of outputs as a tensor [M, B]
        ensemble_tensor = torch.stack(list_outputs)  # [M, B]
        M = ensemble_tensor.size(0)

        # Apply the function v (that is usually set to identity)
        v_ensemble = v_func(ensemble_tensor)     # [M, B]
        v_y = v_func(target).unsqueeze(0)             # [1, B] 

        # Compute the bias
        term1 = torch.mean(torch.abs(v_ensemble - v_y), dim=0)  # [B]

        # Compute the spread over scenarios
        diff = v_ensemble.unsqueeze(1) - v_ensemble.unsqueeze(0)  # [M, M, B]
        term2 = torch.mean(torch.abs(diff), dim=(0,1))  # [B]

        return (term1 - 0.5 * term2).mean()         # mean over the batch
    

    # Compute the mean PITD over the batch (each PITD is computed for the aggregated distribution of the videos & timesteps)
    def PITD_loss(self, list_output, target, dict_pdf, time_step):            
        return PITD().forward(list_output, target, dict_pdf, time_step)

    

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




# Here we conmpute the SSIM, this metric should be computed for each frame (H, W) so we iterate over the batch and the channels then average over it   
# The metric is between 0 and 1 (theoritically it could go to -1 for anticorrelation). The higher the better
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
    


# Compute the rank histogram and the deviation between predictions / target (that should follow a uniform distribution over [0, 1])
# We compute a rank histogram whose base is the data of all the samples (all the timesteps & all the generated videos). One can eventually plot it 
# It returns the average value of the PITD and the pdf of F_X_Y for each quantiles 
class PITD(nn.Module):
    def __init__(self):
        super().__init__()

    # Compute (and eventually plot) the rank histogram for a list of images (C, H, W). The "real" forward is below.
    def compute_rank_histogram(self, list_output, target, plot_histo = False, plot_path = None):      # list_output is a list of predictions (C, H, W). Target is (C, H, W)

        # Concatenate the predictions and flatten into array
        output = torch.cat(list_output, dim=0)                  # (n_scenarios * C, H, W)
        pred_values = np.ravel(output.cpu().detach().numpy())
        target_values = np.ravel(target.cpu().detach().numpy())

        # Sort prediction values once
        pred_sorted = np.sort(pred_values)

        # Compute CDF values: F_X(y) = P(X ≤ y) & F_x_x. In theory, F_X_X should be U[0, 1] but there are some exceptions, thus we prefer to compute the deviation on the "true" F_X_X and not U[0, 1]
        F_X_Y = np.searchsorted(pred_sorted, target_values, side='right') / len(pred_sorted)
        F_X_X = np.searchsorted(pred_sorted, pred_sorted, side='right') / len(pred_sorted)

        # Compute histogram. Get the pre-computed quantiles from the WHOLE training set (See Coméphore/STVD/explore_dataset.py)
        df = pd.read_csv("/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/STVD/Data_analysis/8_quantiles.csv")
        quantiles = np.asarray(df["quantile"])                      # Quantiles of interest, computed from the training set

        # If the target frames are all zeros, then the output is null as well and the predictions are perfect
        if target_values.max() == 0:
            frequency = np.zeros(len(quantiles) - 1)
            frequency[0] = 1 
            return 0, frequency

        # Computes the normalized distribution for the specified quantiles (for the prediction and truth)
        counts, bin_edges = np.histogram(F_X_Y, quantiles)
        size_quantile = [(quantiles[k+1] - quantiles[k]) for k in range(len(counts))]
        counts = counts / size_quantile                # Takes into account the fact quantiles are not equally separated
        frequency = counts / counts.sum()              # Make it a distribution as it sums up to 1

        # We compute it for X as well to plot it, it should follow a uniform distribution
        counts_pred, bin_edges = np.histogram(F_X_X, quantiles)
        counts_pred = counts_pred / size_quantile                   
        frequency_pred = counts_pred / counts_pred.sum()                

        expected_frequency = 1 / len(counts)
        # Plot histogram
        if plot_histo:
            plt.figure(figsize=(8, 5))

            # Compute bin widths
            bin_widths = np.diff(bin_edges)

            # Compute bin centers for x axis
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            # Plot using bar for pred & target
            plt.bar(bin_centers, frequency, width=bin_widths, alpha=0.5, color='g', label = "Target")
            plt.bar(bin_centers, frequency_pred, width=bin_widths, alpha=0.5, color='b', label = "Predictions")
            
            # Plot the reference
            plt.axhline(y = expected_frequency, linestyle="-", color = "r", label = "Uniform distribution")
        
            plt.xlabel("Rank Bin")
            plt.ylabel("Relative Frequency")
            plt.title("Rank Histogram")
            plt.legend()
            plt.savefig(plot_path)

        # Computes the deviation
        pitd = np.sqrt(np.sum(((frequency - frequency_pred) ** 2) * size_quantile))        # We multiple by size_quantile to take into account the fact the quantiles are not equally separated

        return pitd, frequency
    
    # Get the list of scenarios for only one sample out of the list of batch
    def get_sample_from_batch(self, list_output, sample):
        return [list_output[k][sample, :, :, :] for k in range(len(list_output))]
    
    # Plot n samples' histograms from the batch
    def plot_channels(self, list_output, target, plot_path):     # list_output is a list of predictions (B, C, H, W). Target is (B, C, H, W)
        B, C, H, W = target.shape
        os.makedirs(plot_path, exist_ok=True)

        for b in range(min(15, B)):     # We don't plot more than 15 PIT
            local_output = self.get_sample_from_batch(list_output, b)
            local_target = target[b, :, :, :]
            local_plot_path = plot_path + f"sample_{b+1}.png"
            a, c = self.compute_rank_histogram(list_output = local_output,
                                                    target = local_target,
                                                    plot_histo = True,
                                                    plot_path=local_plot_path)
                               
                
    # Compute the PITD for each sample then average it over the batch
    # Fill the array in input with the values at the quantiles
    def forward(self, list_output, target, dict_pdf, time_step):    # list_output is a list of predictions (B, C, H, W). Target is (B, C, H, W). time_step is the list of timesteps of length B

        B, C, H, W = target.shape

        mean_pitd = 0
        for b in range(B):
            local_output = self.get_sample_from_batch(list_output, b)
            local_target = target[b, :, :, :]
            pitd, frequencies = self.compute_rank_histogram(list_output = local_output,
                                                    target = local_target,
                                                    plot_histo = False)
            
            # Stores the quantiles' values
            dict_pdf[time_step[0][b].item()] = frequencies

            mean_pitd += pitd
        
                
        mean_pitd = mean_pitd / B

        return mean_pitd, dict_pdf














