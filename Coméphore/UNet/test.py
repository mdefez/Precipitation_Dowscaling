from ctypes.util import test
import torch
from model import UNet
from dataset import RainSuperResDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import os 
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset
from loss import CustomLoss
import numpy as np

# Data directories
input_dir = '/work/FAC/FGSE/IDYST/tbeucler/downscaling/mdefez/Comephore/RNB/input_data'
output_dir = '/work/FAC/FGSE/IDYST/tbeucler/downscaling/mdefez/Comephore/RNB/target_data'
channel_dir = '/work/FAC/FGSE/IDYST/tbeucler/downscaling/mdefez/Comephore/RNB/input_data/DEM'

# We set the batch size to 1 to compute the loss for each input/ouput, thus we can save the worst/best predictions
batch_size = 1

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
def save_images(input0, input1, predictions, dem, targets, bot_or_top = None, output_dir='/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/UNet/Images'):

    os.makedirs(output_dir, exist_ok=True)

    # We save n examples per epoch
    for i in range(len(predictions)): 
        pred_img = predictions[i].squeeze().cpu().detach().numpy()               # Predictions
        target_img = targets[i].squeeze().cpu().detach().numpy()                 # Targets
        input_0_plot = input0[i].cpu().detach().numpy().squeeze()      # First frame
        input_1_plot = input1[i].cpu().detach().numpy().squeeze()      # Last frame
        dem_plot = dem[i].cpu().detach().numpy().squeeze()             # DEM

        # Useful to organize the plot
        num_channels = pred_img.shape[0]

        # Number of vertical slots
        nb_slots = num_channels + 2         

        plt.figure(figsize=(12, 4 * num_channels))

        # Plot DEM & inputs
        plot_img(dem_plot, False, nb_slots, 1, "DEM")
        plot_img(input_0_plot, True, nb_slots, 3, "First frame")
        plot_img(input_1_plot, True, nb_slots, 4, "Last frame")

        # Loop over the 6 timesteps
        for c in range(num_channels):
            # Prediction
            plot_img(pred_img[c], True, nb_slots, 2 * (c+1) + 3, f"Prediction - Timestep {c+1}")

            # Target
            plot_img(target_img[c], True, nb_slots, 2 * (c+1) + 4, f"Target - Timestep {c+1}")

        # Save the plot
        # Design the name of the file
        if bot_or_top == "bot":
            name_file = f"Lowest {len(predictions) - i} file"
        else:
            name_file = f"Best {i} file"
        plt.savefig(os.path.join(output_dir, f"{output_dir}/{name_file}.png"))
        plt.close()

# Load the model's weights
def load_model(model, filepath):
    checkpoint = torch.load(filepath, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


print("Loading model")

# Filepath to the .pth file, where are stored the model's weights
filepath='/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/UNet/weights/model_cv.pth' # Weights to load

model = UNet()  # Set the type of model we are using
model = load_model(model, filepath)  # Load the weights
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters : {total_params}")


# Load the test dataset
test_dataset = []
for hor in range(4):
    for vert in range(4):
        test_dataset.append(RainSuperResDataset(input_dir, output_dir, channel_dir, hor, vert, train=False))

test_dataset = ConcatDataset(test_dataset)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Evaluate
model.eval()
total_test_loss = 0

base_loss = nn.L1Loss()
lambda_conservative = 0.1
lambda_autocorr = 0.1
criterion = CustomLoss(base_loss=base_loss,
                           lambda_conservative=lambda_conservative,
                           lambda_covariance=lambda_autocorr)

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

with torch.no_grad():
    for inp0, inp1, channel, target in test_loader:
        print(f"Testing progress : {progress*100/n:.3f}%")
        progress += batch_size

        inp0, inp1, channel, target = inp0.to(device), inp1.to(device), channel.to(device), target.to(device)

        output = model(inp0, inp1, channel)  # Compute the output

        test_loss = criterion(output, target).item() # Compute the loss
        total_test_loss += test_loss

        # Update if it si a good/bad sample
        # Fill the list with the first 5 values
        if len(worst_loss) < 5:
            # Worst loss
            worst_loss.append(test_loss)
            worst_sample.append([inp0, inp1, output, channel, target])

            worst_loss, worst_sample = sort_l1_l2(worst_loss, worst_sample) # Both list are sorted ascendingly 

            min_loss_of_the_worst = worst_loss[0]

            # Best loss
            best_loss.append(test_loss)
            best_sample.append([inp0, inp1, output, channel, target])
            best_loss, best_sample = sort_l1_l2(best_loss, best_sample) # Both list are sorted ascendingly 

            max_loss_of_the_best = best_loss[-1]

        # Keep track of the best/worst sample
        # If the list is already filled
        else:
            # If the loss is higher than the min of the 5 highest, we should replace the sample
            if test_loss > min_loss_of_the_worst:
                worst_loss[0] = test_loss
                worst_sample[0] = [inp0, inp1, output, channel, target]
                # Sort both list again & compute the new min of the highest
                worst_loss, worst_sample = sort_l1_l2(worst_loss, worst_sample) # Both list are sorted ascendingly 

                min_loss_of_the_worst = worst_loss[0]

            # Same for the best loss
            if test_loss < max_loss_of_the_best:
                best_loss[-1] = test_loss
                best_sample[-1] = [inp0, inp1, output, channel, target]
                # Sort both list again & compute the new max of the lowest
                best_loss, best_sample = sort_l1_l2(best_loss, best_sample) # Both list are sorted ascendingly 

                max_loss_of_the_best = best_loss[-1]
        

# Plot the best & worst sample
def get_column(worst_sample, col):
    return [x[col] for x in worst_sample]

save_images(get_column(worst_sample, 0), get_column(worst_sample, 1), get_column(worst_sample, 2), get_column(worst_sample, 3), get_column(worst_sample, 4), bot_or_top="bot")
save_images(get_column(best_sample, 0), get_column(best_sample, 1), get_column(best_sample, 2), get_column(best_sample, 3), get_column(best_sample, 4), bot_or_top="top")

avg_test_loss = total_test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss:.4f}")


