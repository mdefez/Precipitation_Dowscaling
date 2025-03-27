import os
import rasterio
import sys
import numpy as np
import re

sys.path.append(os.getcwd())
import Processing_inputs.tools as tool



# The following function groupby hourly file according to the SR temporal factor, which needs to divide 24

def group_rasters_by_time_and_save(directory, temp_factor):
    os.makedirs("Coméphore/Processing_input_data/temporal_downsampling/Images/hourly", exist_ok=True)
    os.makedirs("Coméphore/Processing_input_data/temporal_downsampling/Images/stacked", exist_ok=True)
    # Dictionnary to store the groupped files


    time_groups = {}
    if 24 % temp_factor != 0:
        return "The specified temporal resolution factor does not divide 24"
    
    for k in range(int(24/temp_factor)):
        time_groups[f"{k*temp_factor}-{(k+1)*temp_factor}"] = []

    date = '20190209' # We focus on a specific day
    
    # Find the corresponding files
    for filename in os.listdir(directory):
        if filename.endswith('_RR.gtif') and re.search(date, filename) != None: 
            # Get the hour from the filename
            hour = int(filename[18:20])

            # Load the file and plot it 
            with rasterio.open(os.path.join(directory, filename)) as src:
                tool.plot_coméphore_high_res(gtif_file=os.path.join(directory, filename), 
                                              output_folder="Coméphore/Processing_input_data/temporal_downsampling/Images/hourly")
                data = src.read(1)
                list_keys = list(time_groups.keys()) 
                list_int_keys = [int(list_keys[k].split("-")[0]) for k in range(int(24/temp_factor))]

                # We find the right folder to add it given the hour
                for k in range(int(24/temp_factor)-1):
                    if hour < list_int_keys[k+1] and hour >= list_int_keys[k]:
                        time_groups[f"{k*temp_factor}-{(k+1)*temp_factor}"].append(data)
                if hour >= list_int_keys[-1]:
                    time_groups[f"{list_int_keys[-1]}-24"].append(data)


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
                tool.plot_coméphore_high_res(gtif_file=output_path, 
                                             output_folder="Coméphore/Processing_input_data/temporal_downsampling/Images/stacked",
                                             nb_hours=temp_factor)

                print(f"Uploaded file : {output_filename}")

# Appeler la fonction avec ton répertoire
directory = "../../../downscaling/mdefez/Comephore/Projected_data/test/9829/2019/COMEPHORE_2019_2/2019"
group_rasters_by_time_and_save(directory, 4)


