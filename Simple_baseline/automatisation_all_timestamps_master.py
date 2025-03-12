import subprocess
import pickle
import os 
import xarray as xr

# Acquire ERA-5 dataset
ds = xr.open_dataset("../Data/ERA/ERA5_2019-1_total_precipitation.nc", engine="netcdf4")
ds["tp"] = 1000*ds["tp"] # On passe en mm/h

# Get the CPC file names
nb_files_to_plot = 3
tous_les_cpc =  os.listdir("../Data/CPC_file/")[:nb_files_to_plot]

def temps_a_partir_cpc(cpc): # Extract date and hour from the filename
    chemin_entier = "CPC_file/" + str(cpc)
    ref_fichier = chemin_entier[9:22]
    échantillon_temps = 24*(int(ref_fichier[6:8]) - 1)+ int(ref_fichier[8:10])
    return échantillon_temps

indices_ligne = [temps_a_partir_cpc(cpc) for cpc in tous_les_cpc]

# Temporary file to save, so that we don't have to load the entire ERA-5 dataset each time 
netcdf_file = 'df_temp.nc'

# Loop to run the pipeline for one CPC_file
for cpc, ligne_index in zip(tous_les_cpc, indices_ligne):
    print(cpc)
    # Extract the corresponding ERA data
    ligne = ds.isel(time = ligne_index)

    # Save the corresponding line into a temporary file
    pickle_file = f'ligne_{ligne_index}.pkl'
    with open(pickle_file, 'wb') as f:
        pickle.dump(ligne, f)
    
    # Run the slave with the corresponding cpc_file & ERA line
    result = subprocess.run(['python', 'Simple_baseline/automatisation_all_timestamps_slave.py', "CPC_file/" + str(cpc), pickle_file], capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    # Delete the temporary file
    os.remove(pickle_file)