import subprocess
import pickle
import os 
import xarray as xr

month_to_work_on = 1

# Get the COMEPHORE file names, we only test on february for the moment
tous_les_com =  os.listdir(f"../../../downscaling/mdefez/Comephore/Projected_data/2019/COMEPHORE_2019_{month_to_work_on}/2019")

nb_files_to_plot = 1
tous_les_com = tous_les_com[:nb_files_to_plot]

# If we want to plot a specific date
one = True
if one == True:
    tous_les_com = ["Projected_2019020918_RR.gtif"]


def temps_a_partir_cpc(cpc): # Extract date and hour from the filename
    ref_fichier = cpc
    échantillon_temps = 24*(int(ref_fichier[16:18]) - 1)+ int(ref_fichier[18:20]) 
    return échantillon_temps


indices_ligne = [temps_a_partir_cpc(com) for com in tous_les_com]


# Loop to run the pipeline for one com_file
for com, ligne_index in zip(tous_les_com, indices_ligne):
    print(com)



    
    # Get the name of the com file to give it to the slave
    com_file = f"../../../downscaling/mdefez/Comephore/Projected_data/test/9829/2019/COMEPHORE_2019_{month_to_work_on}/2019/" + com

    # Run the slave with the corresponding cpc_file & ERA line
    result = subprocess.run(['python', 'Coméphore/Simple_baseline_COMEPHORE/automatisation_all_timestamps_slave.py', com, pickle_file, month_to_work_on, com_file], capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
