# The goal of this file is to format correctly the xlsx files containing the filenames and coordinates

import pandas as pd
import re

df_ori = pd.read_excel("/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/DEM/source_copernicus.xlsx")

# We execute this pipeline for each year 
for year in range(2019, 2025):
    print(year)

    # Replace the coordinates column by the true coordinates
    df_ori[f"coordinates {year}"] = df_ori[f"coordinates {year}"].apply(lambda x: x.split(":")[-2][18:29] if pd.isna(x) == False else x)

    # Replace the filename according to the format we need to pass in the API
    df_ori[f"filename to give to the API {year}"] = df_ori[f"filename to give to the API {year}"].apply(lambda x: x.split(".")[0] if pd.isna(x) == False else x)

    # Keep only the GLO-30-DTED rows
    df_ori["to keep"] = df_ori[f"DGED ou DTED {year}"].apply(lambda x: re.search("GLO-30-DTED", x) != None if pd.isna(x) == False else x)
    df_ori = df_ori.loc[df_ori["to keep"] == True].drop(["to keep"], axis = 1)

    # Here we keep the rows corresponding to the tiles of interest
    # Tiles to download for Coméphore
    range_north = range(39, 55)
    range_west = range(1, 11)
    range_east = range(0, 15)

    def keep_tiles(coord):
        if coord[0] == "S":
            return False

        north = int(coord[1:3])
        if north not in range_north:
            return False

        if coord[7] == "W":
            west = int(coord[8:])
            if west not in range_west:
                return False 
        
        else:
            east = int(coord[8:])
            if east not in range_east:
                return False 

        return True

    df_ori["to keep"] = df_ori[f"coordinates {year}"].apply(keep_tiles)
    df_ori = df_ori.loc[df_ori["to keep"] == True].drop(["to keep"], axis = 1)

df_ori.to_excel("/work/FAC/FGSE/IDYST/tbeucler/default/maxdefez/Precipitation_Dowscaling/DEM/correct_format_filename.xlsx")