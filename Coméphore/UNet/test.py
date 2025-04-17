import torch
from baseline import UNet
from UNet_attention import UNet_with_attention
from dataset import RainSuperResDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import os 
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset
from loss import CustomLoss
import numpy as np

# Super resolution factors
temp_factor = 1
spatial_factor = 10
n_inputs = 6        # Frames to take into account as input

name_of_the_run = "cv_hard_attention_l2" + f"spatial_{spatial_factor}_temp_{temp_factor}_input_{n_inputs}"

# Data directories
input_dir = f'/work/FAC/FGSE/IDYST/tbeucler/downscaling/mdefez/Comephore/RNB/input_data/spatial_{spatial_factor}_temp_{temp_factor}'          # Low res frames
output_dir = '/work/FAC/FGSE/IDYST/tbeucler/downscaling/mdefez/Comephore/RNB/target_data'        # High res targets
channel_dir = '/work/FAC/FGSE/IDYST/tbeucler/downscaling/mdefez/Comephore/RNB/input_data/DEM'    # DEM

# We set the batch size to 1 to compute the loss for each input/ouput, thus we can save the worst/best predictions
batch_size = 1

available_strategy_mass = [None, "additive", ("multiplicative", "a function type that operates on tensors")] # The function should apply element wise for tensors
def f_mass(x): # Function to apply element by element to the tensor. Be careful, it should not be zero when x = 0 and i thould not diverge when x is big
    return (x+1)**2

strategy_mass = ("multiplicative", f_mass)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to plot a DEM/Precipitation image
def plot_img(image, is_precip, nb_slot, position, title, nb_column = 2):
    plt.subplot(nb_slot, nb_column, position)

    if is_precip:
        plt.imshow(image, cmap='viridis', vmin = 0, vmax=10)
        plt.colorbar(label = "Precipitation (mm)")
        plt.axis("off")

    else:
        plt.imshow(image, cmap='terrain')
        plt.colorbar(label = "Elevation (m)")

    plt.title(title)

# Function to plot all the relevant images and save them
def save_images(list_input, predictions, dem, targets, bot_or_top = None, 
                output_dir=f'/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/UNet/Images_{name_of_the_run}', sample = None):

    os.makedirs(output_dir, exist_ok=True)

    # We save n examples per epoch
    for i in range(len(predictions)): 
        try:
            pred_img = predictions[i].cpu().detach().numpy()               # Predictions
            target_img = targets[i].cpu().detach().numpy()                 # Targets
            list_input_plot = [inp.cpu().detach().numpy().squeeze() for inp in list_input] # frames
            dem_plot = dem[i].cpu().detach().numpy().squeeze()           # DEM

        except:
            pred_img = predictions[i].cpu().detach().numpy().squeeze(1)             # Predictions
            target_img = targets[i].cpu().detach().numpy().squeeze(1)                 # Targets
            list_input_plot = [inp[0].cpu().detach().numpy().squeeze() for inp in list_input] # frames
            dem_plot = dem[i].cpu().detach().numpy().squeeze()           # DEM

        # Useful to organize the plot
        num_channels = pred_img.shape[0]

        # Number of vertical slots
        nb_slots = num_channels + 1 + len(list_input) // 2          

        plt.figure(figsize=(12, 4 * num_channels))

        # Plot DEM & inputs
        plot_img(dem_plot, False, nb_slots, 1, "DEM")
        k = 0
        for frame in list_input_plot:
            k += 1
            plot_img(frame, True, nb_slots, 1+k, f"Frame {k}")

        # Loop over the 6 timesteps
        for c in range(num_channels):
            # Prediction
            plot_img(pred_img[c], True, nb_slots, 2*(len(list_input)//2) + 2 + 2*c + 1, f"Prediction - Timestep {c+1}")

            # Target
            plot_img(target_img[c], True, nb_slots, 2*(len(list_input)//2) + 2 + 2*c + 2, f"Target - Timestep {c+1}")

        # Save the plot
        # Design the name of the file
        if bot_or_top == "bot":
            name_file = f"Lowest {len(predictions) - i} file"
        elif bot_or_top == "top":
            name_file = f"Best {i} file"
        else:
            name_file = f"Random {sample} file"
        plt.savefig(os.path.join(output_dir, f"{output_dir}/{name_file}.png"))
        plt.close()

# Load the model's weights
def load_model(model, filepath):
    checkpoint = torch.load(filepath, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


print("Loading model")

# Filepath to the .pth file, where are stored the model's weights
filepath=f'/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/UNet/weights/{name_of_the_run}.pth' # Weights to load

model = UNet_with_attention(hard_constraint_mass=strategy_mass, temp_factor=temp_factor, spatial_factor=spatial_factor, n_inputs=n_inputs)  # Set the type of model we are using
model = load_model(model, filepath)  # Load the weights
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters : {total_params}")


# Load the test dataset
test_dataset = []
for hor in range(4):
    for vert in range(4):
        test_dataset.append(RainSuperResDataset(input_dir, output_dir, channel_dir, hor, vert, temp_factor=temp_factor, 
                                                train=False, n_days=5, n_inputs=n_inputs, spatial_factor=spatial_factor))

test_dataset = ConcatDataset(test_dataset)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

# Evaluate
model.eval()
total_test_loss = 0

base_loss = nn.MSELoss()
lambda_conservative = 0.1
lambda_autocorr = 0.1
criterion = CustomLoss(base_loss=base_loss,
                           lambda_conservative=lambda_conservative,
                           lambda_covariance=lambda_autocorr,
                           conservative=False,
                           covariance=False)

# To compute progress
n = test_dataset.__len__()
progress = 0

# To store the best / worst predictions
worst_loss = []
worst_sample = []

best_loss = []
best_sample = []

def sort_l1_l2(l1, l2): # Sort both list according to the first one
    l1 = np.array(l1)
    idx = np.argsort(l1)

    # Réorganisation
    l1_sorted = list(l1[idx])
    l2_sorted = [l2[i] for i in idx]

    return l1_sorted, l2_sorted

plot_first_samples = 1
samples_to_plot = 5

with torch.no_grad():
    for list_low_res, channel, target in test_loader:
        print(f"Testing progress : {progress*100/n:.3f}%")
        progress += batch_size

        channel, target = channel.to(device), target.to(device)
        for k in range(len(list_low_res)):
            list_low_res[k] = list_low_res[k].to(device)

        output = model(list_low_res, channel)     # Compute the output

        test_loss = criterion(output, target).item() # Compute the loss
        total_test_loss += test_loss

        # Plot some random predictions for the first batch
        if plot_first_samples <= samples_to_plot:
            save_images(list_low_res, output, channel, target, sample=plot_first_samples)
            plot_first_samples += 1

        # Update if it is a good/bad sample
        # Fill the list with the first 5 values
        if len(worst_loss) < 5:
            # Worst loss
            worst_loss.append(test_loss)
            worst_sample.append([list_low_res, output, channel, target])

            worst_loss, worst_sample = sort_l1_l2(worst_loss, worst_sample) # Both list are sorted ascendingly 

            min_loss_of_the_worst = worst_loss[0]

            # Best loss
            best_loss.append(test_loss)
            best_sample.append([list_low_res, output, channel, target])
            best_loss, best_sample = sort_l1_l2(best_loss, best_sample) # Both list are sorted ascendingly 

            max_loss_of_the_best = best_loss[-1]

        # Keep track of the best/worst sample
        # If the list is already filled
        else:
            # If the loss is higher than the min of the 5 highest, we should replace the sample
            if test_loss > min_loss_of_the_worst:
                worst_loss[0] = test_loss
                worst_sample[0] = [list_low_res, output, channel, target]
                # Sort both list again & compute the new min of the highest
                worst_loss, worst_sample = sort_l1_l2(worst_loss, worst_sample) # Both list are sorted ascendingly 

                min_loss_of_the_worst = worst_loss[0]

            # Same for the best loss
            if test_loss < max_loss_of_the_best:
                best_loss[-1] = test_loss
                best_sample[-1] = [list_low_res, output, channel, target]
                # Sort both list again & compute the new max of the lowest
                best_loss, best_sample = sort_l1_l2(best_loss, best_sample) # Both list are sorted ascendingly 

                max_loss_of_the_best = best_loss[-1]
        

# Plot the best & worst sample
def get_column(worst_sample, col):
    return [x[col] for x in worst_sample]

save_images(get_column(worst_sample, 0), get_column(worst_sample, 1), get_column(worst_sample, 2), get_column(worst_sample, 3), bot_or_top="bot")
save_images(get_column(best_sample, 0), get_column(best_sample, 1), get_column(best_sample, 2), get_column(best_sample, 3), bot_or_top="top")

avg_test_loss = total_test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss:.4f}")


