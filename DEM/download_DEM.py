# The goal of this file is to download the zip files correspond to the tiles of interest through the Copernicus API

import requests
import pandas as pd
import os 

os.makedirs("../../../downscaling/mdefez/DEM/zip", exist_ok=True) # Where to stock the downloaded zip files

# Put an access token here, be careful it usually resets every 10 minutes
access_token = ""

df_coord_id = pd.read_excel("DEM/coordinates_with_id.xlsx") # Read the df with the product id and corresponding coordinates

for k in range(len(df_coord_id)):

    coord = df_coord_id.loc[k, "coordinates"]
    id = df_coord_id.loc[k, "Id"]

    if pd.isna(id) == False:
        print(f"Downloading : {id}")
        url = f"https://download.dataspace.copernicus.eu/odata/v1/Products({id})/$value" # Replace by the id value, after Products

        headers = {"Authorization": f"Bearer {access_token}"}

        # Create a session and update headers
        session = requests.Session()
        session.headers.update(headers)

        # Perform the GET request
        response = session.get(url, stream=True)

        # Check if the request was successful
        if response.status_code == 200:
            with open(f"../../../downscaling/mdefez/DEM/zip/{coord}.zip", "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive new chunks
                        file.write(chunk)
        else:
            print(f"Failed to download file. Status code: {response.status_code}")
            print(response.text)