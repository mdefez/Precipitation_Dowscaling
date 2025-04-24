import torch
from UNet_attention import UNet_with_attention
from dataset import RainSuperResDataset
from torch.utils.data import DataLoader
import os 
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset
import numpy as np
import tools as tool

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to plot a DEM/Precipitation image
def plot_img(image, is_precip, nb_slot, position, title, nb_column = 2):
    plt.subplot(nb_slot, nb_column, position)
    plt.subplots_adjust(hspace=0.4) 

    if is_precip:
        plt.imshow(image, cmap='viridis', vmin = 0, vmax = 0.1)
        plt.colorbar(label = "Precipitation")
        plt.axis("off")

    else:
        plt.imshow(image, cmap='terrain', vmin = 0, vmax = 1)
        plt.colorbar(label = "Elevation")
        plt.axis("off")

    plt.title(title)

# Function to plot all the relevant images and save them
def save_images(list_input, predictions, dem, targets, output_dir, bot_or_top = None, best_worst = False):

    os.makedirs(output_dir, exist_ok=True)
    for folder in ["Random", "Lowest", "Best"]:
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)
    
    # We save n examples per epoch
    for i in range(min(15, len(predictions))): 

        pred_img = predictions[i].cpu().detach().numpy()               # Predictions
        target_img = targets[i].cpu().detach().numpy()                 # Targets
        if best_worst == False: # If we plot samples from a batch
            list_input_plot = [inp[i].cpu().detach().numpy().squeeze() for inp in list_input] # frames
        if best_worst == True: # If we plot samples from the best/worst samples
            list_input_plot = [inp.cpu().detach().numpy().squeeze() for inp in list_input[i]]
        dem_plot = dem[i].cpu().detach().numpy().squeeze()           # DEM

        # Useful to organize the plot
        num_channels = pred_img.shape[0]

        # Number of vertical slots
        nb_slots = num_channels + 1 + len(list_input_plot) // 2          

        plt.figure(figsize=(6, 5 * num_channels))

        # Plot DEM & inputs
        plot_img(dem_plot, False, nb_slots, 1, "DEM")
        k = 0
        for frame in list_input_plot:
            k += 1
            plot_img(frame, True, nb_slots, 1+k, f"Frame {k}")

        # Loop over the n timesteps
        for c in range(num_channels):
            # Prediction
            plot_img(pred_img[c], True, nb_slots, 2*((len(list_input_plot))//2) + 2 + 2*c + 1, f"Prediction - Timestep {c+1}")

            # Target
            plot_img(target_img[c], True, nb_slots, 2*((len(list_input_plot))//2) + 2 + 2*c + 2, f"Target - Timestep {c+1}")

        # Save the plot
        # Design the name of the file
        if bot_or_top == "bot":
            name_file = f"Lowest/Lowest {len(predictions) - i} file"
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

def test(input_dir, output_dir, channel_dir, 
         spatial_factor, temp_factor, n_inputs, name_of_the_run,
         best_transform, criterion, batch_size, asked_model, model_parameters, n_days):
    
    output_dir_images = f'/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/UNet/Images/spatial_{spatial_factor}_temp_{temp_factor}/{asked_model}/{name_of_the_run}'

    print("Loading model")

    if asked_model == "UNet_with_attention":
        model = UNet_with_attention(model_parameters=model_parameters, temp_factor=temp_factor, spatial_factor=spatial_factor)  # Set the type of model we are using
    


    # Filepath to the .pth file, where are stored the model's weights
    filepath=f'/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/UNet/weights/{name_of_the_run}.pth' # Weights to load

    model = load_model(model, filepath)  # Load the weights
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters : {total_params}")


    # Load the test dataset
    test_dataset = []
    for hor in range(4):
        for vert in range(4):
            test_dataset.append(RainSuperResDataset(input_dir, output_dir, channel_dir, hor, vert, temp_factor=temp_factor, 
                                                    train=False, n_days=n_days, n_inputs=n_inputs, spatial_factor=spatial_factor))

    test_dataset = ConcatDataset(test_dataset)
    # Normalize the test data, according to the training one
    transform_precip, transform_dem = best_transform

    normalized_test_dataset = tool.TransformedDataset(base_dataset = test_dataset,
                                                       transform_precip = transform_precip,
                                                       transform_dem = transform_dem)

    test_loader = DataLoader(normalized_test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    # Evaluate
    model.eval()

    # To compute progress
    n = test_dataset.__len__()
    progress = 0

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
    best_worst_to_plot = 5 # Number of best / worst sample to plot

    with torch.no_grad():
        for list_low_res, channel, target in test_loader:
            print(f"Testing progress : {progress*100/n:.3f}%")
            progress += batch_size

            channel, target = channel.to(device), target.to(device)
            for k in range(len(list_low_res)):
                list_low_res[k] = list_low_res[k].to(device)

            output = model(list_low_res, channel)     # Compute the output

            test_loss = criterion(output, target) # Compute the average loss for each of the specified metric
            loss_vector = criterion.forward_vecteur(output, target) # Compute the marginal loss for each pair of output/target

            try:    # If it exists, add the loss to the total loss
                total_test_loss += test_loss
            except: # If it doesn't exist, initialize it with the first loss
                total_test_loss = test_loss 

            # Plot some random predictions for the first batch
            if plot_first_samples == True:
                save_images(list_low_res, output, channel, target, output_dir=output_dir_images)
                plot_first_samples = False

            # Update if it is a good/bad sample
            # Fill the list with the first 5 values
            if len(worst_loss) < best_worst_to_plot:
                # Worst loss
                for idx in range(best_worst_to_plot - len(worst_loss)): # Fill it until it reaches the right length
                    worst_loss.append(loss_vector[idx].item()) # add the marginal loss
                    worst_sample.append([get_sample(list_low_res, idx), output[idx], channel[idx], target[idx]]) # add the corresponding sample

                worst_loss, worst_sample = sort_l1_l2(worst_loss, worst_sample) # Both list are sorted ascendingly 

                min_loss_of_the_worst = worst_loss[0]

            if len(best_loss) < best_worst_to_plot:
                # Best loss
                for idx in range(best_worst_to_plot - len(best_loss)): # Fill it until it reaches the right length
                    best_loss.append(loss_vector[idx].item()) # add the marginal loss
                    best_sample.append([get_sample(list_low_res, idx), output[idx], channel[idx], target[idx]]) # add the corresponding sample

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
                        worst_sample[0] = [get_sample(list_low_res, k), output[k], channel[k], target[k]]
                        # Sort both list again & compute the new min of the highest
                        worst_loss, worst_sample = sort_l1_l2(worst_loss, worst_sample) # Both list are sorted ascendingly 

                        min_loss_of_the_worst = worst_loss[0]

                    if marginal_loss < max_loss_of_the_best:
                        best_loss[-1] = marginal_loss
                        best_sample[-1] = [get_sample(list_low_res, k), output[k], channel[k], target[k]]
                        # Sort both list again & compute the new min of the highest
                        best_loss, best_sample = sort_l1_l2(best_loss, best_sample) # Both list are sorted ascendingly 

                        max_loss_of_the_best = best_loss[-1]


            

    # Plot the best & worst sample
    def get_column(worst_sample, col):
        return [x[col] for x in worst_sample]

    save_images(get_column(worst_sample, 0), get_column(worst_sample, 1), get_column(worst_sample, 2), get_column(worst_sample, 3),
                 bot_or_top="bot", output_dir=output_dir_images, best_worst=True)
    save_images(get_column(best_sample, 0), get_column(best_sample, 1), get_column(best_sample, 2), get_column(best_sample, 3), 
                bot_or_top="top", output_dir=output_dir_images, best_worst=True)

    avg_test_loss = total_test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss} for the following metrics \n{criterion.name_metric}")


