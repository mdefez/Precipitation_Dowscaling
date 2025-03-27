#######################################
###### BE CAREFUL #####################
# this script is supposed to be ran where you want to download the Coméphore dataset, so probably not here
#######################################

import requests
import os
import tarfile

dataset_id = "669e23a7ce052a9e8521b75e" #ID of the comephore dataset
api_url = f"https://www.data.gouv.fr/api/1/datasets/{dataset_id}/"

# Metadata
response = requests.get(api_url)
data = response.json()

year_to_download = 2019

# Download files
for resource in data["resources"]:
    # Get the year and month
    try:
        année = int(resource["description"][-6:-2])
        mois = int(resource["description"][-2:])
    except ValueError:
        break

    # Only download the 2019 year yet
    if année == year_to_download:

        # Get the url to download
        file_url = resource["url"]

        # Download if doesn't exist yet
        if os.path.exists(f"COMEPHORE_{année}_{mois}.tar") == False:
            print(année, mois)
            file_response = requests.get(file_url)
            with open(f"COMEPHORE_{année}_{mois}.tar", "wb") as f:
                f.write(file_response.content)

# Extract the data wich are .tar file
source = os.getcwd()
dossier_extraction = os.path.join(source, "Original_data")
dossier_extraction = os.path.join(dossier_extraction, str(year_to_download))


os.makedirs(dossier_extraction, exist_ok=True)

for fichier in os.listdir(source):
    if fichier.endswith(".tar"):
        print(f"extracting {fichier}")
        chemin_tar = os.path.join(source, fichier)
        dossier_sortie = os.path.join(dossier_extraction, fichier.replace(".tar", ""))

        os.makedirs(dossier_sortie, exist_ok=True)  # Create file if doesn't exist

        with tarfile.open(chemin_tar, "r") as archive:
            archive.extractall(dossier_sortie) 


# We now can delete the .tar file given that they were exported

dossier_tar = os.getcwd()

for fichier in os.listdir(dossier_tar):
    if fichier.endswith(".tar"):  
        chemin_tar = os.path.join(dossier_tar, fichier)
        print(f"Deleting {fichier}")
        os.remove(chemin_tar)  #delete