# This scripts downloads the original Coméphore dataset into the specified path
# It also extracts the corresponding .tar file

import requests
import os
import tarfile

dataset_id = "669e23a7ce052a9e8521b75e" #ID of the comephore dataset
api_url = f"https://www.data.gouv.fr/api/1/datasets/{dataset_id}/"

# Metadata
response = requests.get(api_url)
data = response.json()

year_to_download = 2023

# Path to the uploading file
path_upload = "../../../downscaling/raw_data/Comephore/Original_data"

# Download files
for resource in data["resources"]:
    # Get the year and month
    try:
        année = int(resource["description"][-6:-2])
        mois = int(resource["description"][-2:])
    except ValueError:
        break

    # Download data for the specified year
    if année == year_to_download:

        # Get the url to download
        file_url = resource["url"]

        # Download if doesn't exist yet
        if os.path.exists(f"{path_upload}/COMEPHORE_{année}_{mois}.tar") == False:
            print(année, mois)
            file_response = requests.get(file_url)
            with open(f"{path_upload}/COMEPHORE_{année}_{mois}.tar", "wb") as f:
                f.write(file_response.content)

# Extract the data wich are .tar file
dossier_extraction = os.path.join(path_upload, str(year_to_download))


os.makedirs(dossier_extraction, exist_ok=True)

for fichier in os.listdir(path_upload):
    if fichier.endswith(".tar"):
        print(f"extracting {fichier}")
        chemin_tar = os.path.join(path_upload, fichier)
        dossier_sortie = os.path.join(dossier_extraction, fichier.replace(".tar", ""))

        os.makedirs(dossier_sortie, exist_ok=True)  # Create file if doesn't exist

        with tarfile.open(chemin_tar, "r") as archive:
            archive.extractall(dossier_sortie) 


# We now can delete the .tar file given that they were exported

for fichier in os.listdir(path_upload):
    if fichier.endswith(".tar"):  
        chemin_tar = os.path.join(path_upload, fichier)
        print(f"Deleting {fichier}")
        os.remove(chemin_tar)  #delete