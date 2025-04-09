import torch
from model import UNet
from dataset import RainSuperResDataset
from torch.utils.data import DataLoader
import os 
import matplotlib.pyplot as plt

# Paramètres
input_dir = '../../../downscaling/mdefez/Comephore/RNB/input_data'
output_dir = '../../../downscaling/mdefez/Comephore/RNB/target_data'
channel_dir = '../../../downscaling/mdefez/Comephore/RNB/input_data/DEM'
batch_size = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Fonction pour enregistrer les images
def save_images(input0, input1, predictions, dem, targets, epoch, output_dir='Coméphore/UNet/Images'):
    os.makedirs(output_dir, exist_ok=True)

    # On veut sauvegarder quelques images par epoch (ex: 3 premières)
    for i in range(min(3, predictions.size(0))):  # Prendre au max 3 images par batch
        # Si l'image a plusieurs canaux (par exemple 3 canaux), on peut afficher chaque canal séparément
        if predictions.dim() == 4:  # (batch_size, channels, height, width)
            pred_img = predictions[i].cpu().detach().numpy()  # Prendre le ième batch
            target_img = targets[i].cpu().detach().numpy()
            input_0_plot = input0[i].cpu().detach().numpy().squeeze()
            input_1_plot = input1[i].cpu().detach().numpy().squeeze()
            dem_plot = dem[i].cpu().detach().numpy().squeeze()

            # Prendre les dimensions du nombre de canaux
            num_channels = pred_img.shape[0]

            nb_slots = num_channels + 2

            # Créer une figure avec le nombre de sous-graphes en fonction du nombre de canaux
            plt.figure(figsize=(12, 4 * num_channels))

            # Plot DEM & inputs
            plot_img(dem_plot, False, nb_slots, 1, "DEM")
            plot_img(input_0_plot, True, nb_slots, 3, "First frame")
            plot_img(input_1_plot, True, nb_slots, 4, "Last frame")

            # Afficher chaque canal séparément
            for c in range(num_channels):
                # Prédiction
                plot_img(pred_img[c], True, nb_slots, 2 * (c+1) + 3, f"Prediction - Timestep {c+1} - Epoch {epoch}")

                # Cible
                plot_img(target_img[c], True, nb_slots, 2 * (c+1) + 4, f"Target - Timestep {c+1} - Epoch {epoch}")

            # Sauvegarder l'image
            plt.savefig(os.path.join(output_dir, f"epoch_{epoch}_sample_{i}.png"))
            plt.close()

# Charger le modèle
def load_model(model, filepath):
    checkpoint = torch.load(filepath)
    # Charge uniquement les poids du modèle, ignore les autres informations
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

# Exemple d'utilisation pour charger le modèle
print("Loading model")
epoch = 0
model = UNet()  # Crée une instance du modèle
model = load_model(model, filepath=f'Coméphore/UNet/model_checkpoint_epoch_{epoch}.pth')  # Charge les poids dans le modèle
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters : {total_params}")
model.to(device)

# Charger le dataset de test
test_dataset = RainSuperResDataset(input_dir, output_dir, channel_dir, train=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

# Évaluation
model.eval()
total_test_loss = 0
criterion = torch.nn.L1Loss()

# To compute progress
n = test_dataset.__len__()
progress = 0

with torch.no_grad():
    plot = True
    for inp0, inp1, channel, target in test_loader:
        print(f"Progress : {progress*100/n:.3f}%")
        progress += batch_size

        inp0, inp1, channel, target = inp0.to(device), inp1.to(device), channel.to(device), target.to(device)

        x = torch.cat([inp0, inp1, channel], dim=1)  # (B, 3, 100, 100)
        output = model(x)                            # (B, 6, 100, 100)

        test_loss = criterion(output, target)
        total_test_loss += test_loss.item()

        if plot == True:
            save_images(inp0, inp1, output, channel, target, epoch="0")
            plot = False

avg_test_loss = total_test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss:.4f}")


