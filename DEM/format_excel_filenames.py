# The goal of this file is to format correctly the xlsx files containing the filenames and coordinates

import pandas as pd
import re

df_ori = pd.read_excel("DEM/source_copernicus.xlsx")

# Replace the coordinates column by the true coordinates
df_ori["coordinates"] = df_ori["coordinates"].apply(lambda x: x.split(":")[-2][18:29])

# Replace the filename according to the format we need to pass in the API
df_ori["filename to give to the API"] = df_ori["filename to give to the API"].apply(lambda x: x.split(".")[0])

# Keep only the GLO-90-DTED rows
df_ori["to keep"] = df_ori["DGED ou DTED"].apply(lambda x: re.search("GLO-30-DTED", x) != None)
df_ori = df_ori.loc[df_ori["to keep"] == True].drop(["to keep"], axis = 1)

# Here we keep the rows corresponding to the tiles of interest
# Tiles to download for Coméphore
range_north = range(39, 55)
range_west = range(170, 180)
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

df_ori["to keep"] = df_ori["coordinates"].apply(keep_tiles)
df_ori = df_ori.loc[df_ori["to keep"] == True].drop(["to keep"], axis = 1)

df_ori.to_excel("DEM/correct_format_filename.xlsx")