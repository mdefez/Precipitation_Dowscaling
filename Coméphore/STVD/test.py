# This file defines a test function
# It takes as input a testing dataset, compute metrics and plots some predictions

import torch
import sys

# Import functions from other files
sys.path.append('/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/Deterministic')
sys.path.append('/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/Diffusion')

from UNet_attention import UNet_with_attention
from diffusion_model import UNetforDiffusion, DiffusionScheduler, TemporalEncoder
from tools_diffu import bicubic_A_seq
from torch.utils.data import DataLoader
import os 
import matplotlib.pyplot as plt
from loss import PITD
import matplotlib.colors as mcolors
from baseline import nearest_neighbor, bicubic
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import numpy as np
from inference import sample_diffusion
from tqdm import tqdm
import pandas as pd
import matplotlib.patches as patches
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to plot a single DEM/Precipitation/Variance image
def plot_img(image, is_precip, nb_slot, position, title, nb_column, delta = False):
    plt.subplot(nb_slot, nb_column, position)
    plt.subplots_adjust(hspace=0.4) 
    plt.subplots_adjust(wspace=0.5) 

    # Custom colormap
    colors = ['white', 'blue', 'yellow']
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('blue_white_red', colors)

    if is_precip == True:
        vmin, vmax = 0, 1
        if delta == True:
            vmin, vmax = -0.1, 0.1
        plt.imshow(image, cmap=custom_cmap, vmin = vmin, vmax = vmax)
        plt.colorbar(label = "Scaled Precip")

        # Remove axis and plot a dark square along the plot
        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.5)
            spine.set_visible(True)

        # Add a star at the max location if there are some precipitation
        if image.max() > 0:
            max_idx = np.unravel_index(np.argmax(image), image.shape)
            max_y, max_x = max_idx  # row = y, column = x
            plt.plot(max_x, max_y, marker='*', color='black', markersize=12, label='Max', linestyle = "None")

    if is_precip == "DEM": # dem
        plt.imshow(image, cmap='terrain', vmin = 0, vmax = 1)
        plt.colorbar(label = "Elevation")
        plt.axis("off")

    if is_precip == "variance": # variance of the scenarios
        custom_cmap = LinearSegmentedColormap.from_list("white_to_red", ["white", "red"])

        plt.imshow(image, cmap=custom_cmap, vmin = 0, vmax = 1)
        plt.colorbar(label = "Standard deviation (mm/h)")
        # Remove axis and plot a dark square along the plot
        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.5)
            spine.set_visible(True)

    plt.title(title, fontsize = 18)

# Function to plot a precip histogram
def plot_histo(pred, target, title, nb_slot, nb_column, position):
    if target.max() == 0:
        return None
    
    plt.subplot(nb_slot, nb_column, position)
    plt.subplots_adjust(hspace=0.4) 
    plt.subplots_adjust(wspace=0.5) 

    flat_pred = pred.flatten()
    flat_target = target.flatten()

    sns.histplot(flat_pred, color='steelblue', label='Prediction', binrange=(0, 1), bins=30, stat='density', alpha =.5)
    sns.histplot(flat_target, color='orange', label='Target', binrange=(0, 1), bins=30, stat='density', alpha = .5)
    plt.title(title, fontsize = 18)
    plt.xlabel("Precipitation")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


# This functions takes a figure as input and adds custom legends at the left of the rows and bottom of the columns to make the plot more readable
def plot_legend_row_columns(fig, num_scenarios, n_rows, num_channels, label_padding = 0.02):

    # Column labels
    n_cols = 1 + num_scenarios + 1 + 1  # Deterministic prediction + scenarios + target + histogram
    column_labels = ["Deterministic prediction"]
    for k in range(num_scenarios):
        column_labels.append(f"Scenario {k+1}")
    if num_scenarios >= 2:      # If we have multiple scenarios, we expect a variance plot
        n_cols += 1
        column_labels.append("Variance")
    column_labels.append("Target")
    column_labels.append("Target & Prediction distribution")

    # Row labels 
    row_labels = [f"Timestep {k}" for k in range(1, num_channels + 1)]
    for k in range(n_rows - num_channels):
        row_labels.insert(0, "Input")

    axes = fig.get_axes()

    # Add custom text at bottom of each column
    begin_horizontal = (axes[0].get_position().x0 + axes[0].get_position().x1) / 2
    horizontal_step = abs(axes[0].get_position().x0 - axes[2].get_position().x0)
    for k in range(n_cols):
        # place text centered under the bottom plot of each column
        fig.text(
            x= begin_horizontal + k*horizontal_step,
            y=0.05,  # near bottom of figure
            s=column_labels[k],
            ha='center',
            va='bottom',
            fontsize = 22,
            color = "#e12020"
        )

    # Add custom text to the left of each row
    begin_vertical = (axes[0].get_position().y0 + axes[0].get_position().y1) / 2
    vertical_step = abs(axes[0].get_position().y0 - axes[-1].get_position().y0) / (n_rows - 1)
    for k in range(n_rows):
        # place text vertically centered beside leftmost plot in the row
        fig.text(
            x=0.05,  # near left edge of figure
            y=begin_vertical - k*vertical_step,
            s=row_labels[k],
            ha='left',
            va='center',
            rotation='vertical',
            fontsize = 22,
            color = "#e12020"
        )

    
    # Add a bar to separate inputs from the rest
    bar_y = begin_vertical - 0.42 * vertical_step
    bar_height = 0.005  # height in figure coordinates

    # Add rectangle across the whole figure
    fig.patches.append(
        patches.Rectangle(
            (0, bar_y),  # (x, y) in figure coordinates
            1.0,         # full figure width
            bar_height,
            transform=fig.transFigure,
            color='black'
        )
    )






# Function to plot all the relevant images of one batch and save them
def save_images(list_input, time_idx, predictions_final, prediction_deter, dem, targets, output_dir, delta, multiple_scenarios, index_folder,
                bot_or_top = None, best_worst = False):

    os.makedirs(output_dir, exist_ok=True)
    # We plot random samples and the best/worst samples (according to the base_loss).
    for folder in [f"Random_{index_folder}", "Lowest", "Best"]:
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)
    
    # We don't plot more than 15 samples
    for i in range(min(15, len(prediction_deter))): 
        if best_worst == False:
            pred_img = []
            for k in range(len(predictions_final)):
                pred_img.append(predictions_final[k][i].cpu().detach().numpy())                 # Final Predictions
        else:
            pred_img = [predictions_final[i].cpu().detach().numpy()]                            # Final Predictions

        pred_img_deter = prediction_deter[i].cpu().detach().numpy()                             # Output (prediction) of the deterministic UNet
        target_img = targets[i].cpu().detach().numpy()                                          # Targets
        dem_plot = dem[i].cpu().detach().numpy().squeeze()                                      # DEM

        # If we plot random samples from a batch
        if best_worst == False: 
            list_input_plot = [inp[i].cpu().detach().numpy().squeeze() for inp in list_input]   # Frames
            list_time = [time[i].cpu().detach().numpy().squeeze() for time in time_idx]         # Corresponding timesteps

        # If we plot best/worst samples
        if best_worst == True: 
            list_input_plot = [inp.cpu().detach().numpy().squeeze() for inp in list_input[i]]   # Frames
            list_time = [time.cpu().detach().numpy().squeeze() for time in time_idx[i]]         # Corresponding timesteps

        # Useful quantities to organize the plot
        num_scenarios = len(pred_img)           # Number of different scenarios to compute
        num_channels = pred_img[0].shape[0]     # Temporal SR factor

        # Number of horizontal slots (columns)
        nb_columns = 1 + num_scenarios + 1 + 1       # Pred deter + n scenarios + target + histogram 
        if multiple_scenarios == True:
            nb_columns += 1                             # Variance

        # Number of vertical slots (rows)
        nb_slots = num_channels + 1 + (len(list_input_plot))// nb_columns     # Input + DEM + temp_factor 

        plt.figure(figsize=(12 + 4 * nb_columns, 5 * num_channels))

        # Plot DEM & inputs on the first row
        plot_img(dem_plot, "DEM", nb_slots, 1, "DEM", nb_column=nb_columns)

        for k in range(len(list_input_plot)):
            plot_img(list_input_plot[k], True, nb_slots, 2+k, f"Frame {k} \n (Timestep {list_time[k]})", nb_column=nb_columns)

        # Loop over the n timesteps
        for c in range(num_channels):
            count = 0           # To keep track of the frame's positionning

            new_scale = False   
            # To change the range of the colormap if we plot deltas
            if delta == True and c >= 1:    
                new_scale = True

            # Prediction of the deterministic model on the first column
            plot_img(pred_img_deter[c], True, nb_slots, nb_columns*((len(list_input_plot))//nb_columns) + nb_columns*(c+1) + count + 1, 
                     f"Prediction (deterministic) \n Timestep {c+1}", delta=new_scale, nb_column=nb_columns)
            count += 1

            # Plot every scenarios
            for k in range(num_scenarios):
                plot_img(pred_img[k][c], True, nb_slots, nb_columns*((len(list_input_plot))//nb_columns) + nb_columns*(c+1) + count + 1 + k, 
                        f"Final prediction \n Timestep {c+1} - Scenario {k+1}", delta=new_scale, nb_column=nb_columns)
            count += 1

            # Plot the variance between scenarios
            if num_scenarios >= 2:
                std = [pred_img[k][c] for k in range(num_scenarios)]
                std = np.stack(std, axis=0)
                std = np.std(std, axis=0)
                max_std = np.max(std)

                # Min max normalization to scale the std to [0, 1]
                if max_std != 0:
                    normalized_std = std / max_std
                else:
                    normalized_std = np.zeros_like(std)
                plot_img(normalized_std, "variance", nb_slots, nb_columns*((len(list_input_plot))//nb_columns) + nb_columns*(c+1) + num_scenarios + count, 
                            f"Scenarios variance \n Timestep {c+1}", delta=new_scale, nb_column=nb_columns)
                count += 1

            # Plot target
            plot_img(target_img[c], True, nb_slots, nb_columns*((len(list_input_plot))//nb_columns) + nb_columns*(c+1) + num_scenarios + count, 
                     f"Target - Timestep {c+1}", delta=new_scale, nb_column=nb_columns)
            count += 1
            
            # Plot the histograms
            average = np.mean(np.stack([pred_img[k][c] for k in range(num_scenarios)], axis = 0), axis = 0) # Take the mean over the scenarios
            target = target_img[c]
            # Plot the target and predicted histogram over the same figure
            plot_histo(pred = average, target = target, title = f"Distribution - Timestep {c+1}", nb_slot=nb_slots, nb_column=nb_columns,
                        position=nb_columns*((len(list_input_plot))//nb_columns) + nb_columns*(c+1) + num_scenarios + count)
            count += 1
                           

        # Make the plot more readable
        fig = plt.gcf()  # Get current figure
        plot_legend_row_columns(fig = fig, num_scenarios = num_scenarios, n_rows = nb_slots, num_channels = num_channels)

        # Save the plot
        # Design the name of the file
        if bot_or_top == "bot":
            name_file = f"Lowest/Lowest {len(predictions_final) - i} file"
        elif bot_or_top == "top":
            name_file = f"Best/Best {i + 1} file"
        else:
            name_file = f"Random_{index_folder}/Random {i + 1} file"


        plt.savefig(os.path.join(output_dir, f"{output_dir}/{name_file}.png"), bbox_inches = "tight")
        plt.close()


# Function to load the model's weights
def load_model(model, filepath):
    checkpoint = torch.load(filepath, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

# Main function to test the model
def test(test_dataset, spatial_factor, temp_factor, name_of_the_run, n_scenarios, use_diffusion,
         criterion, batch_size, asked_model, model_parameters, delta, n_inputs, model_parameters_diffusion, start_time):
    
    # Folder where one should save the plots
    output_dir_images = f'/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/STVD/Images/spatial_{spatial_factor}_temp_{temp_factor}/{asked_model}/{name_of_the_run}'

    print("Loading deterministic model")

    # Load the deterministic model
    if asked_model == "UNet_with_attention":
        model = UNet_with_attention(model_parameters=model_parameters, temp_factor=temp_factor, spatial_factor=spatial_factor)  
    
    elif asked_model == "bicubic":
        model_deter = bicubic(temp_factor=temp_factor, spatial_factor=spatial_factor)
        model_deter.to(device)

    elif asked_model == "nearest_neighbor":
        model_deter = nearest_neighbor(temp_factor=temp_factor, spatial_factor=spatial_factor)
        model_deter.to(device)

    # Filepath to the .pth file, where are stored the model's weights
    filepath_deter = f'/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/Deterministic/weights/spatial_{spatial_factor}_temp_{temp_factor}/{name_of_the_run}.pth' # Weights to load
    filepath_diffu = f'/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/Diffusion/weights/spatial_{spatial_factor}_temp_{temp_factor}/{name_of_the_run}.pth' # Weights to load


    if asked_model not in ["bicubic", "nearest_neighbor"]: # If the model is trainable, load the weights
        model_deter = load_model(model, filepath_deter)  # Load the weights
        model_deter.to(device)

    # Load the diffusion model
    print("Loading diffusion model")
    in_channels = 2*(temp_factor) + 1
    nb_steps, beta, conservative_mass_diffusion, nb_heads, window_size, strat_attention_diffu = model_parameters_diffusion

    if use_diffusion:
        # Load the model
        model_diffu = UNetforDiffusion(in_channels=in_channels, base_channels=64, embed_dim=256, time_emb_dim = 128, temp_factor = temp_factor, 
                                    spatial_factor = spatial_factor, window_size = window_size, nb_heads = nb_heads, strat_attention = strat_attention_diffu)
        model_diffu = load_model(model_diffu, filepath_diffu)  
        model_diffu.to(device)

        # Load the scheduler & temporal encoder
        scheduler = DiffusionScheduler(timesteps=nb_steps, beta_start=beta[0], beta_end=beta[1], type = beta[2])
        temporal_encoder = TemporalEncoder(input_channels=1, embed_dim=256, seq_len=n_inputs).to(device).eval()

    # Load the test dataset
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True) 

    # Evaluate
    model_deter.eval()
    if use_diffusion:
        model_diffu.eval()


    # To store the best / worst samples
    worst_loss, worst_sample = [], []
    best_loss, best_sample = [], []

    # Get the first n samples of the batch from a list of inputs (list of batch)
    def get_first_samples(list_frames, nb):  
        samples = []
        for i in range(len(list_frames)):
            new_idx = list_frames[i][:nb]
            samples.append(new_idx)

        return samples
    
    # Get the sample corresponding to the specified idx of the batch from a list of inputs (list of batch)
    def get_sample(list_frames, idx):  
        samples = []
        for i in range(len(list_frames)):
            new_idx = list_frames[i][idx]
            samples.append(new_idx)

        return samples

    # Sort both lists according to the first one 
    def sort_l1_l2(l1, l2): 
        l1 = np.array(l1)
        idx = np.argsort(l1)

        l1_sorted = list(l1[idx])
        l2_sorted = [l2[i] for i in idx]

        return l1_sorted, l2_sorted
    
    # Compute the average PITD from the dictionnary and plot it
    def average_PITD(dict_pdf, plot_path):
        # Convert the dictionnary of quantiles to a 2D array
        values = np.array(list(dict_pdf.values()))  # shape: (num_lists, list_length)

        # Compute the mean across rows (i.e., average per index)
        average_array = np.mean(values, axis=0)

        # Get the quantiles
        df = pd.read_csv("/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/STVD/Data_analysis/8_quantiles.csv")
        quantiles = np.asarray(df["quantile"])                      # Quantiles of interest, computed from the training set

        plt.figure(figsize=(8, 5))

        # Compute bin widths
        bin_widths = np.diff(quantiles)

        # Compute bin centers for x axis
        bin_centers = 0.5 * (quantiles[:-1] + quantiles[1:])

        # Plot using bar for pred & target
        plt.bar(bin_centers, average_array, width=bin_widths, alpha=0.5, color='g', label = "Average PITD")
        
        # Plot the reference
        expected_frequency = 1 / len(quantiles)
        plt.axhline(y = expected_frequency, linestyle="-", color = "r", label = "Uniform distribution")
    
        plt.xlabel("Rank Bin")
        plt.ylabel("Relative Frequency")
        plt.title("Rank Histogram")
        plt.legend()
        plt.savefig(plot_path)

        # Test
        size_quantile = [(quantiles[k+1] - quantiles[k]) for k in range(len(average_array))]
        pitd = np.sqrt(np.sum(((average_array - expected_frequency) ** 2) * size_quantile))

    plot_first_samples = 0      
    batch_to_plot = 3           # How many batch we want to plot
    best_worst_to_plot = 5      # Number of best / worst samples to plot

    with torch.no_grad():

        crps = 0                        # Initialize the CRPS value
        PITD_metric = 0
        ploted_sample_pitd = False      # Keep track of the sample PITD plot, we plot them only for the first batch
        dict_pdf = {}                   # This dictionnary stores the values of the pdf (of each sample) that helps building each PIT plot. The goal is to compute an "average" PIT for the whole dataset and evaluate the model calibration

        Time_limit = 2.8 * 24 * 60 * 60  # If the code runs for more than 2.7 days, break it and return the temporary results
        multiple_scenarios = True if n_scenarios >=2 else False

        # Loop over the testing set
        for list_low_res, channel, target, time_idx in tqdm(test_loader, desc="Testing"):

            # Compute the output of the deterministic model
            output_deter = model_deter(list_low_res, channel, apply_constraint = True)     

            if use_diffusion:
                # Compute the output of the diffusion model
                A_seq = bicubic_A_seq(list_low_res)     # Compute the bicubic HR from the LR to pass the diffusion UNet

                # The output is a list of n "real" outputs. It corresponds to n different scenarios
                # We only compute the n scenarios to plot them (and compute CRPS). After the first batch, we set n to 1. 
                B_pred = sample_diffusion(model_diffu, scheduler, A_seq, C = output_deter, temporal_encoder = temporal_encoder, n_scenarios = n_scenarios,
                                        num_steps=nb_steps, last_frame = list_low_res[-1], conservative_mass_diffusion = conservative_mass_diffusion,
                                        device = device) 
                
            else:       # If one only uses the deterministic model, we compute one scenario and put it into a list to feed the right format
                B_pred = [output_deter]

            # We compute every loss for a random scenario (the first, so that it works when we have only one scenario)
            test_loss = criterion(B_pred[0], target) 

            # Plot PIT and compute marginals PITD (only for the first batch)
            if ploted_sample_pitd == False:
                PITD().plot_channels(list_output = B_pred, target = target, 
                               plot_path = f"/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/STVD/Images/spatial_{spatial_factor}_temp_{temp_factor}/{asked_model}/{name_of_the_run}/PITD/")
                ploted_sample_pitd = True

            # Compute CRPS & PITD (only if we use a probabilistic approach ie the diffusion model)
            PITD_marginal, dict_pdf = criterion.PITD_loss(B_pred, target, dict_pdf, time_idx)                 # Compute the mean PITD over the batch. Read loss.py for more details
            PITD_metric += PITD_marginal
            if use_diffusion:   
                crps += criterion.crps(B_pred, target, lambda x: x)         # Compute the mean CRPS over the batch, the function sets the weights in the CRPS formula
            
            loss_vector = criterion.forward_vecteur(B_pred[0], target)          # Compute the marginal loss only for the main metric

            # Update the test loss
            try:    
                total_test_loss += test_loss
            except: 
                total_test_loss = test_loss 

            # Plot some random predictions only for the first batch
            if plot_first_samples < batch_to_plot:
                save_images(list_low_res, time_idx, B_pred, output_deter, channel, target, output_dir=output_dir_images, index_folder=plot_first_samples, 
                            delta = delta, multiple_scenarios = multiple_scenarios, best_worst = False)
                plot_first_samples += 1

            ### Keep track of the best/worst samples ###
            # Fill the list with the first 5 values
            if len(worst_loss) < best_worst_to_plot:
                # Worst loss
                for idx in range(best_worst_to_plot - len(worst_loss)):         # Fill it until it reaches the right length
                    worst_loss.append(loss_vector[idx].item())                  # Add the marginal loss
                    worst_sample.append([get_sample(list_low_res, idx), get_sample(time_idx, idx), B_pred[0][idx], output_deter[idx], channel[idx], target[idx]]) # add the corresponding sample

                worst_loss, worst_sample = sort_l1_l2(worst_loss, worst_sample) # Both list are sorted ascendingly 

                min_loss_of_the_worst = worst_loss[0]

            if len(best_loss) < best_worst_to_plot:
                # Best loss
                for idx in range(best_worst_to_plot - len(best_loss)):                          # Fill it until it reaches the right length
                    best_loss.append(loss_vector[idx].item())                   # add the marginal loss
                    best_sample.append([get_sample(list_low_res, idx), get_sample(time_idx, idx), B_pred[0][idx], output_deter[idx], channel[idx], target[idx]]) # add the corresponding sample

                best_loss, best_sample = sort_l1_l2(best_loss, best_sample)         # Both list are sorted ascendingly 

                max_loss_of_the_best = best_loss[-1]


            # If the list is already filled
            else:
                for k in range(len(loss_vector)):
                    marginal_loss = loss_vector[k].item()
                    # If the loss is higher than the min of the n highest, we should replace the sample
                    if marginal_loss > min_loss_of_the_worst:
                        worst_loss[0] = marginal_loss
                        worst_sample[0] = [get_sample(list_low_res, k), get_sample(time_idx, k), B_pred[0][k], output_deter[k], channel[k], target[k]]

                        # Sort both list again & compute the new min of the highest
                        worst_loss, worst_sample = sort_l1_l2(worst_loss, worst_sample) # Both list are sorted ascendingly 

                        min_loss_of_the_worst = worst_loss[0]

                    # If the loss is lower than the max of the n lowest, we should replace the sample
                    if marginal_loss < max_loss_of_the_best:
                        best_loss[-1] = marginal_loss
                        best_sample[-1] = [get_sample(list_low_res, k), get_sample(time_idx, k), B_pred[0][k], output_deter[k], channel[k], target[k]]
                        # Sort both list again & compute the new min of the highest
                        best_loss, best_sample = sort_l1_l2(best_loss, best_sample) # Both list are sorted ascendingly 

                        max_loss_of_the_best = best_loss[-1]
            
            # Interrupt if running for too long
            if time.time() - start_time > Time_limit:
                print(f"Time limit of {Time_limit / (24*60*60)} days reached")
                break




            

    # Useful functions to get a specific column of a list of list
    def get_column(sample, col):
        return [x[col] for x in sample]

    # Plot the best & worst samples
    save_images(get_column(worst_sample, 0), get_column(worst_sample, 1), get_column(worst_sample, 2), get_column(worst_sample, 3), get_column(worst_sample, 4),
                get_column(worst_sample, 5), bot_or_top="bot", output_dir=output_dir_images, best_worst=True, delta = delta, multiple_scenarios = False, index_folder = None)
    save_images(get_column(best_sample, 0), get_column(best_sample, 1), get_column(best_sample, 2), get_column(best_sample, 3), get_column(best_sample, 4),
                get_column(best_sample, 5), bot_or_top="top", output_dir=output_dir_images, best_worst=True, delta = delta, multiple_scenarios = False, index_folder = None)

    # Print the metrics
    avg_test_loss = total_test_loss / len(test_loader)
    crps = crps / len(test_loader)
    PITD_metric = PITD_metric / len(test_loader)

    # Plot the average PITD
    average_PITD(dict_pdf, f"/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/STVD/Images/spatial_{spatial_factor}_temp_{temp_factor}/{asked_model}/{name_of_the_run}/PITD/average.png")

    print(f"Test Loss: {avg_test_loss} for the following metrics \n{criterion.name_metric}")
    print(f"Mean PITD = {PITD_metric}")
    if use_diffusion:
        print(f"Mean CRPS = {crps}")
        


