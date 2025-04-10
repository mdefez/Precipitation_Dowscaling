# The goal of this script is to decompressed the gtif file & downsample it to a 1km scale

import tarfile
import rasterio
import numpy as np
from rasterio.enums import Resampling

### Decompressing the file
input_folder = "/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/DEM_open_topo/rasters_COP90.tar.gz"
output_folder = "/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/DEM_open_topo"

with tarfile.open(input_folder, "r:gz") as archive:
    archive.extractall(path=output_folder)

### Downsampling it 

input_file = "/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/DEM_open_topo/output_hh.tif"
output_file = "/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/DEM_open_topo/low_res.tif"

# Résolution cible en degrés (par exemple 0.01° par pixel)
target_resolution_deg = 0.011376662713563385  # en degrés (pour obtenir environ 1 km à l'équateur)


with rasterio.open(input_file) as src:
    # Récupérer les informations de géotransformation et de résolution
    src_res_x, src_res_y = src.res
    src_transform = src.transform

    # Calculer les nouvelles dimensions du raster en fonction de la nouvelle résolution en degrés
    new_width = int((src.bounds[2] - src.bounds[0]) / target_resolution_deg)  # en x (longitude)
    new_height = int((src.bounds[3] - src.bounds[1]) / target_resolution_deg)  # en y (latitude)

    # Effectuer le reéchantillonnage à la résolution cible
    # Pour le reéchantillonnage, on va utiliser l'option 'average' qui correspond à la moyenne des pixels
    data = src.read(1)  # Lire la première bande
    downsampled_data = np.zeros((new_height, new_width), dtype=np.float32)

    # Utiliser Rasterio pour reéchantillonner l'image avec une moyenne
    # Nous utilisons 'Resampling.average' pour calculer la moyenne des pixels dans chaque cellule du nouveau raster
    downsampled_data = src.read(
        1,
        out_shape=(new_height, new_width),  # dimensions du nouveau raster
        resampling=Resampling.average  # Appliquer la moyenne
    )

    # Créer la nouvelle transformation affine pour la géotransformation
    new_transform = rasterio.transform.from_origin(
        src.bounds[0],  # longitude min (coin supérieur gauche)
        src.bounds[3],  # latitude max (coin supérieur gauche)
        target_resolution_deg,  # nouvelle résolution en degrés pour chaque pixel
        target_resolution_deg  # même résolution pour l'axe y
    )

    # Nouveau profil pour le fichier de sortie
    profile = src.profile
    profile.update({
        'transform': new_transform,
        'width': new_width,
        'height': new_height,
        'count': 1  # Mono-bande
    })

    # Créer le fichier de sortie avec les données reéchantillonnées
    with rasterio.open(output_file, 'w', **profile) as dst:
        dst.write(downsampled_data, 1)
        print(dst.res)

        







