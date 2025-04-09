import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from model import UNet
from dataset import RainSuperResDataset
import os 
import matplotlib.pyplot as plt

# Paramètres d'entraînement
input_dir = '../../../downscaling/mdefez/Comephore/RNB/input_data'
output_dir = '../../../downscaling/mdefez/Comephore/RNB/target_data'
channel_dir = '../../../downscaling/mdefez/Comephore/RNB/input_data/DEM'
batch_size = 20
epochs = 5
learning_rate = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Fonction pour sauvegarder les poids du modèle
def save_model(model, epoch, optimizer, filepath='Coméphore/UNet/'):
    # Enregistrer l'état du modèle
    filename = f"model_checkpoint_epoch_{epoch}.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath + filename)


# Charger le dataset
print("Data loading")
train_dataset = RainSuperResDataset(input_dir, output_dir, channel_dir, train=True)
test_dataset = RainSuperResDataset(input_dir, output_dir, channel_dir, train=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
print("Data loaded")

# Initialiser le modèle, la fonction de perte et l'optimiseur
model = UNet().to(device)
criterion = nn.MSELoss()  # MAE
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Entraînement
print("Training")
for epoch in range(epochs):
    print(f"Epoch {epoch}")
    model.train()
    total_loss = 0

    # To compute progress
    progress = 0 
    n = train_dataset.__len__()
    
    for inp0, inp1, channel, target in train_loader:
        print(f"Training progress : {100*progress/n}%")
        progress += batch_size

        inp0, inp1, channel, target = inp0.to(device), inp1.to(device), channel.to(device), target.to(device)

        x = torch.cat([inp0, inp1, channel], dim=1)  # (B, 3, 100, 100)
        output = model(x)                            # (B, 6, 100, 100)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    print("Saving model")
    save_model(model, epoch, optimizer)

    # Test toutes les n = 1 epoch
    model.eval()
    total_test_loss = 0

    # To compute progresss
    k = 0 
    n = train_dataset.__len__()
    with torch.no_grad():
        for inp0, inp1, channel, target in test_loader:
            print(f"Testing progress : {100*k/n}%")
            k += batch_size 

            inp0, inp1, channel, target = inp0.to(device), inp1.to(device), channel.to(device), target.to(device)

            x = torch.cat([inp0, inp1, channel], dim=1)  # (B, 3, 100, 100)
            output = model(x)                            # (B, 6, 100, 100)

            test_loss = criterion(output, target)
            total_test_loss += test_loss.item()

    avg_test_loss = total_test_loss / len(test_loader)
    print(f"Test Loss after Epoch {epoch+1}: {avg_test_loss:.4f}")


