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
import matplotlib.colors as mcolors
from baseline import nearest_neighbor, bicubic
import numpy as np
from tqdm import tqdm
from inference import sample_diffusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to plot a DEM/Precipitation image
def plot_img(image, is_precip, nb_slot, position, title, nb_column, delta = False):
    plt.subplot(nb_slot, nb_column, position)
    plt.subplots_adjust(hspace=0.4) 
    plt.subplots_adjust(wspace=0.5) 

    # Custom colormap
    colors = ['white', 'blue', 'yellow']
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('blue_white_red', colors)

    if is_precip:
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

    else: # dem
        plt.imshow(image, cmap='terrain', vmin = 0, vmax = 1)
        plt.colorbar(label = "Elevation")
        plt.axis("off")

    plt.title(title)
    

# Function to plot all the relevant images and save them
def save_images(list_input, time_idx, predictions_final, prediction_deter, dem, targets, output_dir, delta, multiple_scenarios,
                bot_or_top = None, best_worst = False):

    os.makedirs(output_dir, exist_ok=True)
    for folder in ["Random", "Lowest", "Best"]:
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)
    
    # We save n examples per epoch
    for i in range(min(15, len(prediction_deter))): 
        if multiple_scenarios == True:
            pred_img = []
            for k in range(len(predictions_final)):
                pred_img.append(predictions_final[k][i].cpu().detach().numpy())               # Final Predictions
        else:
            pred_img = [predictions_final[i].cpu().detach().numpy()]               # Final Predictions
        pred_img_deter = prediction_deter[i].cpu().detach().numpy()               # Output (prediction) of the deterministic UNet
        target_img = targets[i].cpu().detach().numpy()                 # Targets
        if best_worst == False: # If we plot samples from a batch
            list_input_plot = [inp[i].cpu().detach().numpy().squeeze() for inp in list_input] # frames
            list_time = [time[i].cpu().detach().numpy().squeeze() for time in time_idx]     # corresponding timesteps
        if best_worst == True: # If we plot samples from the best/worst samples
            list_input_plot = [inp.cpu().detach().numpy().squeeze() for inp in list_input[i]]
            list_time = [time.cpu().detach().numpy().squeeze() for time in time_idx[i]]
        dem_plot = dem[i].cpu().detach().numpy().squeeze()           # DEM

        # Useful to organize the plot
        num_scenarios = len(pred_img)

        num_channels = pred_img[0].shape[0]

        # Number of horizontal slots
        nb_columns = 1 + num_scenarios + 1      # Pred deter + n scenarios + target

        # Number of vertical slots
        nb_slots = num_channels + 1 + (len(list_input_plot))// nb_columns     # DEM + input + temp_factor (given that we have 3 columns)      

        plt.figure(figsize=(12 + 4 * nb_columns, 5 * num_channels))

        # Plot DEM & inputs
        plot_img(dem_plot, False, nb_slots, 1, "DEM", nb_column=nb_columns)

        for k in range(len(list_input_plot)):
            plot_img(list_input_plot[k], True, nb_slots, 2+k, f"Frame {k} (Timestep {list_time[k]})", nb_column=nb_columns)

        # Loop over the n timesteps
        for c in range(num_channels):
            new_scale = False   
            if delta == True and c >= 1:    # To change the range of the colormap if we plot deltas
                new_scale = True

            # Prediction of the deterministic model
            plot_img(pred_img_deter[c], True, nb_slots, nb_columns*((len(list_input_plot))//nb_columns) + nb_columns*(c+1) + 1, 
                     f"Prediction (deterministic) - Timestep {c+1}", delta=new_scale, nb_column=nb_columns)

            for k in range(num_scenarios):
                # Plot a scenario
                plot_img(pred_img[k][c], True, nb_slots, nb_columns*((len(list_input_plot))//nb_columns) + nb_columns*(c+1) + 2 + k, 
                        f"Final prediction - Timestep {c+1} - Scenario {k+1}", delta=new_scale, nb_column=nb_columns)

            # Target
            plot_img(target_img[c], True, nb_slots, nb_columns*((len(list_input_plot))//nb_columns) + nb_columns*(c+1) + 2 + num_scenarios, 
                     f"Target - Timestep {c+1}", delta=new_scale, nb_column=nb_columns)

        # Save the plot
        # Design the name of the file
        if bot_or_top == "bot":
            name_file = f"Lowest/Lowest {len(predictions_final) - i} file"
        elif bot_or_top == "top":
            name_file = f"Best/Best {i + 1} file"
        else:
            name_file = f"Random/Random {i + 1} file"

        plt.savefig(os.path.join(output_dir, f"{output_dir}/{name_file}.png"))
        plt.close()

# Load the model's weights
def load_model(model, filepath):
    checkpoint = torch.load(filepath, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def test(test_dataset, spatial_factor, temp_factor, name_of_the_run, n_scenarios,
         criterion, batch_size, asked_model, model_parameters, delta, n_inputs, model_parameters_diffusion):
    
    output_dir_images = f'/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/STVD/Images/spatial_{spatial_factor}_temp_{temp_factor}/{asked_model}/{name_of_the_run}'

    print("Loading deterministic model")

    if asked_model == "UNet_with_attention":
        model = UNet_with_attention(model_parameters=model_parameters, temp_factor=temp_factor, spatial_factor=spatial_factor)  # Set the type of model we are using
    
    elif asked_model == "bicubic":
        model = bicubic(temp_factor=temp_factor, spatial_factor=spatial_factor)

    elif asked_model == "nearest_neighbor":
        model = nearest_neighbor(temp_factor=temp_factor, spatial_factor=spatial_factor)

    # Filepath to the .pth file, where are stored the model's weights
    filepath_deter = f'/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/Deterministic/weights/spatial_{spatial_factor}_temp_{temp_factor}/{name_of_the_run}.pth' # Weights to load
    filepath_diffu = f'/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/Diffusion/weights/spatial_{spatial_factor}_temp_{temp_factor}/{name_of_the_run}.pth' # Weights to load


    if asked_model not in ["bicubic", "nearest_neighbor"]: # If the model is trainable, load the weights
        model_deter = load_model(model, filepath_deter)  # Load the weights
        model_deter.to(device)

    print("Loading diffusion model")
    in_channels = 2*(temp_factor) + 1
    nb_steps, beta, conservative_mass_diffusion = model_parameters_diffusion

    model_diffu = UNetforDiffusion(in_channels=in_channels, base_channels=64, embed_dim=256, time_emb_dim = 128, temp_factor = temp_factor, 
                                   spatial_factor = spatial_factor)
    model_diffu = load_model(model_diffu, filepath_diffu)  # Load the weights
    model_diffu.to(device)

    scheduler = DiffusionScheduler(timesteps=nb_steps, beta_start=beta[0], beta_end=beta[1], type = beta[2])
    temporal_encoder = TemporalEncoder(input_channels=1, embed_dim=256, seq_len=n_inputs).to(device).eval()

    # Loading the test dataset

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    # Evaluate
    model_deter.eval()
    model_diffu.eval()


    # To store the best / worst predictions
    worst_loss = []
    worst_sample = []

    best_loss = []
    best_sample = []

    def get_first_samples(list_frames, nb):  # Get the first n samples of the batch from a list of inputs (list of batch)
        samples = []
        for i in range(len(list_frames)):
            new_idx = list_frames[i][:nb]
            samples.append(new_idx)

        return samples
    
    def get_sample(list_frames, idx):  # Get the sample corresponding to the specified idx of the batch from a list of inputs (list of batch)
        samples = []
        for i in range(len(list_frames)):
            new_idx = list_frames[i][idx]
            samples.append(new_idx)

        return samples

    def sort_l1_l2(l1, l2): # Sort both lists according to the first one 
        l1 = np.array(l1)
        idx = np.argsort(l1)

        # Réorganisation
        l1_sorted = list(l1[idx])
        l2_sorted = [l2[i] for i in idx]

        return l1_sorted, l2_sorted

    plot_first_samples = True
    best_worst_to_plot = 10 # Number of best / worst sample to plot

    with torch.no_grad():
        stop = 10
        count = 0
        for list_low_res, channel, target, time_idx in test_loader:
            count += 1
            if count == stop:
                break
            channel, target = channel.to(device), target.to(device)
            for k in range(len(list_low_res)):
                list_low_res[k] = list_low_res[k].to(device)

            # Compute the output of the deterministic model
            output_deter = model_deter(list_low_res, channel, apply_constraint = True)     

            # Compute the output of the diffusion model
            A_seq = bicubic_A_seq(list_low_res)     # Compute the HR from the LR to pass the diffusion UNet

            # The output is a list of n "real" outputs. It corresponds to n different scenarios
            # We only compute the n scenarios to plot them. After the first batch, we set n to 1. 
            B_pred = sample_diffusion(model_diffu, scheduler, A_seq, C = output_deter, temporal_encoder = temporal_encoder, n_scenarios = n_scenarios,
                                      num_steps=nb_steps, last_frame = list_low_res[-1], conservative_mass_diffusion = conservative_mass_diffusion) 

            # We compute the loss for a random scenario (the first for example)
            test_loss = criterion(B_pred[0], target) # Compute the average loss for each of the specified metric
            loss_vector = criterion.forward_vecteur(B_pred[0], target) # Compute the marginal loss only for the main metric

            try:    # If it exists, add the loss to the total loss
                total_test_loss += test_loss
            except: # If it doesn't exist, initialize the total loss with the first loss
                total_test_loss = test_loss 

            # Plot some random predictions (from the deterministic only AND for the whole model) for the first batch
            if plot_first_samples == True:
                save_images(list_low_res, time_idx, B_pred, output_deter, channel, target, output_dir=output_dir_images, delta = delta, multiple_scenarios = True)
                plot_first_samples = False
                n_scenarios = 1

            # Update if it is a good/bad sample
            # Fill the list with the first 5 values
            if len(worst_loss) < best_worst_to_plot:
                # Worst loss
                for idx in range(best_worst_to_plot - len(worst_loss)): # Fill it until it reaches the right length
                    worst_loss.append(loss_vector[idx].item()) # add the marginal loss
                    worst_sample.append([get_sample(list_low_res, idx), get_sample(time_idx, idx), B_pred[0][idx], output_deter[idx], channel[idx], target[idx]]) # add the corresponding sample

                worst_loss, worst_sample = sort_l1_l2(worst_loss, worst_sample) # Both list are sorted ascendingly 

                min_loss_of_the_worst = worst_loss[0]


            threshold_best = 0.01
            if len(best_loss) < best_worst_to_plot:
                # Best loss
                idx = 0
                while len(best_loss) < best_worst_to_plot: # Fill it until it reaches the right length
                    if get_sample(list_low_res, idx)[-1].mean() > threshold_best:   # Fill it with valid (non null) candidates
                        best_loss.append(loss_vector[idx].item()) # add the marginal loss
                        best_sample.append([get_sample(list_low_res, idx), get_sample(time_idx, idx), B_pred[0][idx], output_deter[idx], channel[idx], target[idx]]) # add the corresponding sample
                    idx += 1

                best_loss, best_sample = sort_l1_l2(best_loss, best_sample) # Both list are sorted ascendingly 

                max_loss_of_the_best = best_loss[-1]



            # Keep track of the best/worst sample
            # If the list is already filled
            else:
                # If the loss is higher than the min of the 5 highest, we should replace the sample
                for k in range(len(loss_vector)):
                    marginal_loss = loss_vector[k].item()
                    if marginal_loss > min_loss_of_the_worst:
                        worst_loss[0] = marginal_loss
                        worst_sample[0] = [get_sample(list_low_res, k), get_sample(time_idx, k), B_pred[0][k], output_deter[k], channel[k], target[k]]

                        # Sort both list again & compute the new min of the highest
                        worst_loss, worst_sample = sort_l1_l2(worst_loss, worst_sample) # Both list are sorted ascendingly 

                        min_loss_of_the_worst = worst_loss[0]

                    last_frame = get_sample(list_low_res, k)[-1]
                    if last_frame.mean() > threshold_best: # For "best" plots, we only consider those with at least some precipitation
                        if marginal_loss < max_loss_of_the_best:
                            best_loss[-1] = marginal_loss
                            best_sample[-1] = [get_sample(list_low_res, k), get_sample(time_idx, k), B_pred[0][k], output_deter[k], channel[k], target[k]]
                            # Sort both list again & compute the new min of the highest
                            best_loss, best_sample = sort_l1_l2(best_loss, best_sample) # Both list are sorted ascendingly 

                            max_loss_of_the_best = best_loss[-1]




            

    # Plot the best & worst sample
    def get_column(worst_sample, col):
        return [x[col] for x in worst_sample]

    save_images(get_column(worst_sample, 0), get_column(worst_sample, 1), get_column(worst_sample, 2), get_column(worst_sample, 3), get_column(worst_sample, 4),
                get_column(worst_sample, 5), bot_or_top="bot", output_dir=output_dir_images, best_worst=True, delta = delta, multiple_scenarios = False)
    save_images(get_column(best_sample, 0), get_column(best_sample, 1), get_column(best_sample, 2), get_column(best_sample, 3), get_column(best_sample, 4),
                get_column(best_sample, 5), bot_or_top="top", output_dir=output_dir_images, best_worst=True, delta = delta, multiple_scenarios = False)

    avg_test_loss = total_test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss} for the following metrics \n{criterion.name_metric}")


