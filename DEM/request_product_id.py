# The goal of this file is to extract the product id corresponding to each specified tiles

import pandas as pd
import requests

df_coordinates = pd.read_excel("DEM/correct_format_filename.xlsx")

list_product_id = []

# We try to request the id according to the filename of each year, running decreasingly trhough the years
for k in range(len(df_coordinates)):
    print(f"Progress : {100*k / len(df_coordinates)}%")

    found = False
    for year in range(2024, 2018, -1):
        filename = df_coordinates.loc[k, f"filename to give to the API {year}"]

        req = requests.get(f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Name eq '{filename}'")
        json = req.json()
        df_temp = pd.DataFrame.from_dict(json['value'])

        if "Id" in df_temp.columns:
            found = True
            list_product_id.append(df_temp["Id"].loc[0])
            break
    if found == False:
        list_product_id.append(None)


df_coordinates["Id"] = list_product_id

df_coordinates.to_excel("DEM/coordinates_with_id.xlsx")

