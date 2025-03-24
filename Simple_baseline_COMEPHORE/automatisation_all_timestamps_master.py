import subprocess
import pickle
import os 
import xarray as xr

# Acquire ERA-5 dataset
ds = xr.open_dataset("../../raw_data/ECMWF/ERA5/SL/total_precipitation/ERA5_2019-1_total_precipitation.nc", engine="netcdf4")
ds["tp"] = 1000*ds["tp"] # On passe en mm/h

# Get the COMEPHORE file names, we only test on january for the moment
tous_les_com =  os.listdir("../../../downscaling/raw_data/Comephore/Projected_data/2019/COMEPHORE_2019_1/2019")


nb_files_to_plot = 1
tous_les_com = tous_les_com[:nb_files_to_plot]

def temps_a_partir_cpc(cpc): # Extract date and hour from the filename
    ref_fichier = cpc
    échantillon_temps = 24*(int(ref_fichier[16:18]) - 1)+ int(ref_fichier[18:20])
    return échantillon_temps

indices_ligne = [temps_a_partir_cpc(com) for com in tous_les_com]

# Temporary file to save, so that we don't have to load the entire ERA-5 dataset each time 
netcdf_file = 'df_temp.nc'

# Loop to run the pipeline for one CPC_file
for com, ligne_index in zip(tous_les_com, indices_ligne):
    print(com)
    # Extract the corresponding ERA data
    ligne = ds.isel(time = ligne_index)

    # Save the corresponding line into a temporary file
    pickle_file = f'ligne_{ligne_index}.pkl'
    with open(pickle_file, 'wb') as f:
        pickle.dump(ligne, f)
    
    # Run the slave with the corresponding cpc_file & ERA line
    result = subprocess.run(['python', 'Simple_baseline_COMEPHORE/automatisation_all_timestamps_slave.py', com, pickle_file], capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    # Delete the temporary file
    os.remove(pickle_file)