import subprocess
import pickle
import os 
import xarray as xr

# Récupérer le dataset ERA
ds = xr.open_dataset("../Data/ERA/ERA5_2019-1_total_precipitation.nc", engine="netcdf4")
ds["tp"] = 1000*ds["tp"] # On passe en mm/h

# Définir une liste de valeurs que tu veux passer à ton script
tous_les_cpc =  os.listdir("../Data/CPC_file/")

def temps_a_partir_cpc(cpc): # Extrait la date et l'heure à partir du nom du fichier
    chemin_entier = "CPC_file/" + str(cpc)
    ref_fichier = chemin_entier[9:22]
    échantillon_temps = 24*(int(ref_fichier[6:8]) - 1)+ int(ref_fichier[8:10])
    return échantillon_temps

indices_ligne = [temps_a_partir_cpc(cpc) for cpc in tous_les_cpc]

# Fichier NetCDF
netcdf_file = 'df_temp.nc'

# Boucle pour exécuter le script avec différentes valeurs et lignes spécifiques
for cpc, ligne_index in zip(tous_les_cpc, indices_ligne):
    print(cpc)
    # Extraire la ligne du fichier NetCDF
    ligne = ds.isel(time = ligne_index)

    # Sauvegarder la ligne extraite dans un fichier pickle temporaire
    pickle_file = f'ligne_{ligne_index}.pkl'
    with open(pickle_file, 'wb') as f:
        pickle.dump(ligne, f)
    
    # Appeler le script_a_executer.py avec la valeur et la ligne extraite
    result = subprocess.run(['python', 'Simple_baseline/automatisation_all_timestamps_slave.py', "CPC_file/" + str(cpc), pickle_file], capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    # Supprimer le fichier pickle après utilisation
    os.remove(pickle_file)