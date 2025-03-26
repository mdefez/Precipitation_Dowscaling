import os
import rasterio
import pandas as pd
import numpy as np
from rasterio.merge import merge
from datetime import datetime


# The following function groupby hourly file according to the SR temporal factor, which needs to divide 24

def group_rasters_by_time_and_save(directory):
    # Dictionnaire pour stocker les rasters regroupés
    time_groups = {'00-06': [], '06-12': [], '12-18': [], '18-00': []}
    date = '20190209' # We focus on a specific day
    # Parcourir tous les fichiers .gtif dans le répertoire
    for filename in os.listdir(directory):
        if filename.endswith('_RR.gtif') and filename.startswith(date): 
            print(filename)
            # Extraire la date et l'heure du nom de fichier
            hour = int(filename[8:10])

            # Charger le fichier raster
            with rasterio.open(os.path.join(directory, filename)) as src:
                if hour >= 0 and hour < 6:
                    time_groups['00-06'].append(src.read(1))
                elif hour >= 6 and hour < 12:
                    time_groups['06-12'].append(src.read(1))
                elif hour >= 12 and hour < 18:
                    time_groups['12-18'].append(src.read(1))
                elif hour >= 18 and hour < 24:
                    time_groups['18-00'].append(src.read(1))

    # Somme des rasters dans chaque groupe et sauvegarde les fichiers

    for time_period, rasters in time_groups.items():
        if rasters:  # Si le groupe contient des rasters
            summed_raster = np.sum(rasters, axis=0)
            
            # Créer un nom pour le fichier de sortie basé sur la date et le groupe horaire
            output_filename = f"{date}_{time_period}.gtif"
            output_path = os.path.join(directory, output_filename)

            # Sauvegarder le raster agrégé dans un nouveau fichier
            with rasterio.open(os.path.join(directory, filename)) as src:
                meta = src.meta
                meta.update(dtype=rasterio.float32, count=1, driver='GTiff')  # Spécifier le driver 'GTiff'
                
                # Sauvegarder le nouveau fichier
                with rasterio.open(output_path, 'w', **meta) as dst:
                    dst.write(summed_raster.astype(rasterio.float32), 1)
                
                print(f"Fichier sauvegardé : {output_filename}")

# Appeler la fonction avec ton répertoire
directory = "../../../downscaling/raw_data/Comephore/Original_data/2019/COMEPHORE_2019_2/2019"  # Remplace par ton chemin
group_rasters_by_time_and_save(directory)


# Supprimer les fichiers générés car ils ne doivent pas être dans original data
def delete_files_with_dash(directory):
    # Parcours tous les fichiers du répertoire
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        # Vérifie si le nom du fichier contient un tiret ("-")
        if "-" in filename:
            os.remove(file_path)
            print(f"Fichier supprimé : {filename}")

# Exemple d'utilisation
directory = "../../../downscaling/raw_data/Comephore/Original_data/2019/COMEPHORE_2019_2/2019"  # Remplace par le chemin de ton répertoire
delete_files_with_dash(directory)
