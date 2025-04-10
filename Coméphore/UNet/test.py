import torch
from model import UNet
from dataset import RainSuperResDataset
from torch.utils.data import DataLoader
import os 
import matplotlib.pyplot as plt

# Data directories
input_dir = '/work/FAC/FGSE/IDYST/tbeucler/downscaling/mdefez/Comephore/RNB/input_data'
output_dir = '/work/FAC/FGSE/IDYST/tbeucler/downscaling/mdefez/Comephore/RNB/target_data'
channel_dir = '/work/FAC/FGSE/IDYST/tbeucler/downscaling/mdefez/Comephore/RNB/input_data/DEM'
batch_size = 32
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
def save_images(input0, input1, predictions, dem, targets, epoch, output_dir='/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/UNet/Images'):
    os.makedirs(os.path.join(output_dir, f"epoch_{epoch}"), exist_ok=True)

    # We save 3 examples per epoch
    for i in range(min(3, predictions.size(0))): 
        # We consider the i-th item of the batch
        pred_img = predictions[i].cpu().detach().numpy()               # Predictions
        target_img = targets[i].cpu().detach().numpy()                 # Targets
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
            plot_img(pred_img[c], True, nb_slots, 2 * (c+1) + 3, f"Prediction - Timestep {c+1} - Epoch {epoch}")

            # Target
            plot_img(target_img[c], True, nb_slots, 2 * (c+1) + 4, f"Target - Timestep {c+1} - Epoch {epoch}")

        # Save the plot
        plt.savefig(os.path.join(output_dir, f"epoch_{epoch}/sample_{i}.png"))
        plt.close()

# Load the model's weights
def load_model(model, filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


print("Loading model")
epoch = 5       # Epoch from where we should test the model
model = UNet()  # Set the type of model we are using
model = load_model(model, filepath=f'/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/Coméphore/UNet/weights/model_checkpoint_epoch_{epoch}.pth')  # Load the weights
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters : {total_params}")


# Load the test dataset
test_dataset = RainSuperResDataset(input_dir, output_dir, channel_dir, train=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

# Evaluate
model.eval()
total_test_loss = 0
criterion = torch.nn.MSELoss()

# To compute progress
n = test_dataset.__len__()
progress = 0

with torch.no_grad():
    plot = True
    for inp0, inp1, channel, target in test_loader:
        print(f"Testing progress : {progress*100/n:.3f}%")
        progress += batch_size

        inp0, inp1, channel, target = inp0.to(device), inp1.to(device), channel.to(device), target.to(device)

        output = model(inp0, inp1, channel)  # Compute the output

        test_loss = criterion(output, target) # Compute the loss
        total_test_loss += test_loss.item()

        if plot == True: # If this is the first batch, plot some predictions
            save_images(inp0, inp1, output, channel, target, epoch)
            plot = False

avg_test_loss = total_test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss:.4f}")


