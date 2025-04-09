# The goal of this file is to download the zip files correspond to the tiles of interest through the Copernicus API

import requests
import pandas as pd
import os 

os.makedirs("../../../downscaling/mdefez/DEM/zip", exist_ok=True) # Where to stock the downloaded zip files

# Put an access token here, be careful it usually resets every 10 minutes
access_token = "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJYVUh3VWZKaHVDVWo0X3k4ZF8xM0hxWXBYMFdwdDd2anhob2FPLUxzREZFIn0.eyJleHAiOjE3NDQwMzEzNjAsImlhdCI6MTc0NDAzMDc2MCwianRpIjoiMDgyNGEwYWItMTE3OS00Yjc2LWE1NzAtZGQzYWQxZGI1OTI0IiwiaXNzIjoiaHR0cHM6Ly9pZGVudGl0eS5kYXRhc3BhY2UuY29wZXJuaWN1cy5ldS9hdXRoL3JlYWxtcy9DRFNFIiwiYXVkIjpbIkNMT1VERkVSUk9fUFVCTElDIiwiYWNjb3VudCJdLCJzdWIiOiIyNzI1Y2Y5OS05MGIzLTQzNjctYmIyNS01YjUwODRiYjIyNjQiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJjZHNlLXB1YmxpYyIsInNlc3Npb25fc3RhdGUiOiI1Nzc3YmZkOS1hMTZlLTQ5MGYtYWI2Yy1hNzRlNmQ0OWQwMDYiLCJhbGxvd2VkLW9yaWdpbnMiOlsiaHR0cHM6Ly9sb2NhbGhvc3Q6NDIwMCIsIioiLCJodHRwczovL3dvcmtzcGFjZS5zdGFnaW5nLWNkc2UtZGF0YS1leHBsb3Jlci5hcHBzLnN0YWdpbmcuaW50cmEuY2xvdWRmZXJyby5jb20iXSwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iLCJkZWZhdWx0LXJvbGVzLWNkYXMiLCJjb3Blcm5pY3VzLWdlbmVyYWwiXX0sInJlc291cmNlX2FjY2VzcyI6eyJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6IkFVRElFTkNFX1BVQkxJQyBvcGVuaWQgZW1haWwgcHJvZmlsZSBvbmRlbWFuZF9wcm9jZXNzaW5nIHVzZXItY29udGV4dCIsInNpZCI6IjU3NzdiZmQ5LWExNmUtNDkwZi1hYjZjLWE3NGU2ZDQ5ZDAwNiIsImdyb3VwX21lbWJlcnNoaXAiOlsiL2FjY2Vzc19ncm91cHMvdXNlcl90eXBvbG9neS9jb3Blcm5pY3VzX2dlbmVyYWwiLCIvb3JnYW5pemF0aW9ucy9kZWZhdWx0LTI3MjVjZjk5LTkwYjMtNDM2Ny1iYjI1LTViNTA4NGJiMjI2NC9yZWd1bGFyX3VzZXIiXSwiZW1haWxfdmVyaWZpZWQiOnRydWUsIm5hbWUiOiJNYXggRGVmZXoiLCJvcmdhbml6YXRpb25zIjpbImRlZmF1bHQtMjcyNWNmOTktOTBiMy00MzY3LWJiMjUtNWI1MDg0YmIyMjY0Il0sInVzZXJfY29udGV4dF9pZCI6IjdkZmQzYzE4LTQ5MGMtNDcwYS1hNzg2LTNiNmYxOGQ5ODE5ZCIsImNvbnRleHRfcm9sZXMiOnt9LCJjb250ZXh0X2dyb3VwcyI6WyIvYWNjZXNzX2dyb3Vwcy91c2VyX3R5cG9sb2d5L2NvcGVybmljdXNfZ2VuZXJhbC8iLCIvb3JnYW5pemF0aW9ucy9kZWZhdWx0LTI3MjVjZjk5LTkwYjMtNDM2Ny1iYjI1LTViNTA4NGJiMjI2NC9yZWd1bGFyX3VzZXIvIl0sInByZWZlcnJlZF91c2VybmFtZSI6Im1heC5kZWZlekBzdHVkZW50LWNzLmZyIiwiZ2l2ZW5fbmFtZSI6Ik1heCIsImZhbWlseV9uYW1lIjoiRGVmZXoiLCJ1c2VyX2NvbnRleHQiOiJkZWZhdWx0LTI3MjVjZjk5LTkwYjMtNDM2Ny1iYjI1LTViNTA4NGJiMjI2NCIsImVtYWlsIjoibWF4LmRlZmV6QHN0dWRlbnQtY3MuZnIifQ.P1FhD_a5i2fz2wn4oJwp2Y5j01huG9hhpxOM2VqKxS-e8mnfzPvNXlH57JNuR7WFjVWRAjRWRP633oVG_8-HwTLGFMmuLzYOOi4mmYCyKf5jhe8FE_mD1JOZZ6e_iqLi3bo_09F4Y52VKgVtFztAK52_v84aQyCMXtIgmgNx0y05idJ7rJ1aX-OU5idW68qOhcTGEOrpTTEDFh6qEWfJgkrAFQK_vPvtT4uT3bOP331meUsyrimr6oZ4eu-jbo_5Siho1S2KeUtBJQM76SbZ0eyOaZ1yjSwJNA8oNINBCwcdhIySTW5fve3nWR2piwIUVRxvBkAi3VllHktu2on0Tw"


df_coord_id = pd.read_excel("DEM/coordinates_with_id.xlsx") # Read the df with the product id and corresponding coordinates

n = len(df_coord_id)
df_coord_id = df_coord_id.iloc[n//2:] # We divide it in two parts because the token expires and we don't have the time to download the whole df

for k in range(len(df_coord_id)):
    k += n//2 # To uncomment
    coord = df_coord_id.loc[k, "coordinates 2019"]
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