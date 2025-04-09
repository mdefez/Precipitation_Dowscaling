import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class RainSuperResDataset(Dataset):
    def __init__(self, input_root, output_root, channel_root, train=True):
        self.samples = []
        self.input_root = input_root
        self.output_root = output_root
        self.channel_root = channel_root

        # Sélectionner les années pour l'entraînement et le test
        if train:
            years = ['2023']  # Utiliser les données de 2023 pour l'entraînement
        else:
            years = ['2024']  # Utiliser les données de 2024 pour le test

        # Liste des domaines à charger
        domains = os.listdir(os.path.join(input_root, years[0]))

        # Créer les échantillons pour chaque domaine et chaque année
        for year in years:
            for domain in domains:
                input_files = sorted(os.listdir(os.path.join(input_root, year, domain))) # Timesteps
                input_times = [int(f[10:20]) for f in input_files]
                input_times.sort()

                # Créer paires consécutives espacées de 6h
                for i in range(len(input_times) - 1): # If there is a 6 hours gap
                    t0 = input_times[i]
                    t1 = input_times[i + 1]
                    if t1 - t0 == 6:  # Add the couple to the list of samples
                        self.samples.append({
                            "year": year,
                            "domain": domain,
                            "t0": t0,
                            "t1": t1
                        })

    def __len__(self):
        return len(self.samples)
    
    def input_format(self, timestep): # Return the correct input filename corresponding to the timestep
        return f"beggining_{timestep}_temp_factor_6_spatial_factor_25.npy"
    
    def output_format(self, timestep): # Return the correct output filename corresponding to the timestep
        return f"{timestep}.npy"
    
    def dem_name(self, domain): # Return the correct dem filename corresponding to the domain
        hor = domain[9]
        vert = domain[16]
        return f"dem_hor_{hor}_vert_{vert}.npy"

    def __getitem__(self, idx): # Return (image_timestep_t, image_timestep_t+1, dem, target) where images are input upsampled through bicubic
        sample = self.samples[idx]
        year = sample["year"]
        domain = sample["domain"]
        t0 = sample["t0"]
        t1 = sample["t1"]

        # --- Input basse résolution ---
        inp0 = np.load(os.path.join(self.input_root, year, domain, self.input_format(t0)))
        inp1 = np.load(os.path.join(self.input_root, year, domain, self.input_format(t1)))

        inp0 = torch.tensor(inp0).unsqueeze(0).unsqueeze(0).float()
        inp1 = torch.tensor(inp1).unsqueeze(0).unsqueeze(0).float()

        inp0_up = F.interpolate(inp0, size=(100, 100), mode='bicubic', align_corners=False)
        inp1_up = F.interpolate(inp1, size=(100, 100), mode='bicubic', align_corners=False)

        # --- Channel du domaine ---
        channel = np.load(os.path.join(self.channel_root, self.dem_name(domain)))
        channel = torch.tensor(channel).unsqueeze(0).float()  # (1, 100, 100)

        # --- Cibles haute résolution ---
        targets = []
        for t in range(t0, t1):  # Inclure t0 jusqu'à t1 - 1 
            target_path = os.path.join(self.output_root, year, domain, self.output_format(t))
            target = np.load(target_path)
            targets.append(torch.tensor(target).unsqueeze(0).float())
        targets = torch.stack(targets)  # (6, 1, 100, 100)
        targets = targets.squeeze(1)

        return inp0_up.squeeze(0), inp1_up.squeeze(0), channel, targets
