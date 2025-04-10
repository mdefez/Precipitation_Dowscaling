# The goal of this file is to dezip the compressed files acquired through the Copernicus API

import zipfile
import os

os.makedirs("/work/FAC/FGSE/IDYST/tbeucler/downscaling/mdefez/DEM/data", exist_ok=True) # Create a folder to store the decompressed files

all_zip = os.listdir("/work/FAC/FGSE/IDYST/tbeucler/downscaling/mdefez/DEM/zip") # List all the compressed files

for zip in all_zip:
    print(f"Decompressing : {zip}")
    zip_file_path = os.path.join("/work/FAC/FGSE/IDYST/tbeucler/downscaling/mdefez/DEM/zip", zip)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref: # Decompress the file
        name = zip.split(".")[0]
        zip_ref.extractall(f"/work/FAC/FGSE/IDYST/tbeucler/downscaling/mdefez/DEM/data/{name}")



