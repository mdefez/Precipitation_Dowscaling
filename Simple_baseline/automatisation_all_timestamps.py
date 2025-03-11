import sys
import pickle
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import subprocess
from scipy.ndimage import zoom
import numpy as np
from scipy.interpolate import griddata
from sklearn.neighbors import KNeighborsRegressor
import h5py
import pyproj
import pandas as pd
import seaborn as sns
from scipy.stats import wasserstein_distance
from scipy.stats import ks_2samp
from libpysal.weights import W
from pysal.explore import esda
from scipy.stats import gaussian_kde
import pikepdf


# Cette fonction lance toute la pipeline : calcul des cartes/matrices puis des métriques
def main(cpc_file, ds):
    ref_fichier = cpc_file[9:22]
    date = f" {cpc_file[15:17]} January 2019 {cpc_file[17:19]}H"

    chemin_image = os.path.join(os.getcwd(), "Images")
    fichier = os.path.join(chemin_image, ref_fichier)
    if not os.path.exists(fichier):
        os.makedirs(fichier)  # Créer le dossier si il n'existe pas

    with PdfPages(f"Images/{ref_fichier}/figures.pdf") as pdf_fig:

        ########## Chargement des données ERA-5 ##########

        vmin = 0
        vmax = 4

        def plot_map(df_plot, nom):
            # Créer une figure et un axe avec la projection PlateCarree
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})

            # Ajouter les côtes et les frontières
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black')

            # Ajouter les frontières de la Suisse avec un fond
            ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')

            # Définir l'étendue géographique pour zoomer sur la Suisse
            # ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())  # Longitude min, Longitude max, Latitude min, Latitude max

            # Tracer les précipitations sur la carte avec la palette conditionnée sur la Suisse
            df_plot.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='viridis', 
                                cbar_kwargs={'label': "Precipitation during the passed hour (mm)"}, 
                                vmin=vmin, vmax=vmax)

            # Ajouter une grille
            ax.gridlines(draw_labels=True, linestyle = ":", linewidth = .5)

            titre = nom.split("/")[-1][:-4] + "\n" + date
            ax.set_title(titre)

            # Afficher la carte
            pdf_fig.savefig(dpi = 100)
            plt.close()
            

        precip = ds.copy()

        # Définir la zone de la Suisse en tenant compte des coordonnées 0-360 pour la longitude
        lon_min_output = 3.168779677002355 
        lat_min_output = 43.6290303456092 
        lon_max_output = 12.46232838782734 
        lat_max_output = 49.36326405028229

        # Utiliser .where() pour filtrer les données dans cette région (filtrage flexible)
        precip_suisse = precip.where(
            (precip['latitude'] >= lat_min_output) & (precip['latitude'] <= lat_max_output) &
            (precip['longitude'] >= lon_min_output) & (precip['longitude'] <= lon_max_output), drop=True
        )

        #precip_mean = ds["tp"].sum(dim="time")
        plot_map(precip_suisse["tp"], f"Images/{ref_fichier}/Basse résolution.png")

        ########## Chargement des données Interpolation bicubique ###################################################################



        # Charger ton dataset ERA5
        data_bicubic = precip_suisse.copy()

        # Obtenir les valeurs de latitude et longitude du dataset
        latitudes = data_bicubic['latitude'].values
        longitudes = data_bicubic['longitude'].values
        precipitation = data_bicubic['tp'].values  

        # Utiliser np.linspace pour générer un nombre spécifique de points entre les bornes min et max
        new_latitudes = np.linspace(latitudes.min(), latitudes.max(), 640)
        new_longitudes = np.linspace(longitudes.min(), longitudes.max(), 710)

        # Créer une grille de nouvelles coordonnées
        new_lon, new_lat = np.meshgrid(new_longitudes, new_latitudes)

        # Créer un maillage des coordonnées d'origine
        lon, lat = np.meshgrid(longitudes, latitudes)

        # Effectuer l'interpolation avec `griddata` de Scipy (méthode cubic pour bicubique)
        precipitation_fine = griddata(
            (lon.flatten(), lat.flatten()), 
            precipitation.flatten(), 
            (new_lon, new_lat), 
            method='cubic'
        )

        # Créer un DataArray avec les nouvelles coordonnées
        ds_interp = xr.DataArray(precipitation_fine, coords=[('latitude', new_latitudes), ('longitude', new_longitudes)])

        plot_map(ds_interp, f"Images/{ref_fichier}/Interpolation bicubique.png")

        # On inverse l'axe des y de la matrice (voir OneNote pourquoi)
        ds_interp_iso = np.array(ds_interp)[::-1, :]

        ########## Chargement des données KNN ###################################################################



        # Charger ton dataset ERA5
        data_bicubic = precip_suisse.copy()

        # Obtenir les valeurs de latitude et longitude du dataset
        latitudes = data_bicubic['latitude'].values
        longitudes = data_bicubic['longitude'].values
        precipitation = data_bicubic['tp'].values  

        # Utiliser np.linspace pour générer un nombre spécifique de points entre les bornes min et max
        new_latitudes = np.linspace(latitudes.min(), latitudes.max(), 640)
        new_longitudes = np.linspace(longitudes.min(), longitudes.max(), 710)

        # Créer une grille de nouvelles coordonnées
        new_lon, new_lat = np.meshgrid(new_longitudes, new_latitudes)

        # Créer un maillage des coordonnées d'origine
        lon, lat = np.meshgrid(longitudes, latitudes)

        # Aplatir les matrices pour utiliser dans le modèle KNeighborsRegressor
        coords = np.vstack([lat.flatten(), lon.flatten()]).T
        new_coords = np.vstack([new_lat.flatten(), new_lon.flatten()]).T

        # Créer et entraîner un modèle KNeighborsRegressor
        n_voisin = 5
        knn = KNeighborsRegressor(n_neighbors=n_voisin, weights="distance")  # Utilisation d'un seul voisin (pour "nearest neighbor")
        knn.fit(coords, precipitation.flatten())

        # Faire les prédictions pour la grille fine
        precipitation_fine = knn.predict(new_coords).reshape(new_latitudes.shape[0], new_longitudes.shape[0])

        # Créer un DataArray avec les nouvelles coordonnées
        ds_knn = xr.DataArray(precipitation_fine, coords=[('latitude', new_latitudes), ('longitude', new_longitudes)])

        plot_map(ds_knn, f"Images/{ref_fichier}/Plus proche voisin k = {n_voisin}.png")
        ds_knn_iso = np.array(ds_knn)[::-1, :]

        ########## Chargement des données CPC ###################################################################

        # Ouvrir le fichier h5
        with h5py.File(cpc_file, 'r') as file:
            df = file["/dataset1/data1/data"][:]



        # Ouvre le fichier HDF5 en mode lecture
        with h5py.File(cpc_file, 'r') as f:
            # Affiche la structure du fichier

            df = f["/dataset1"]["data1"]["data"]
            df = pd.DataFrame(df[:])  

            # Définir les systèmes de coordonnées
            epsg_2056 = pyproj.CRS("EPSG:2056")
            epsg_4326 = pyproj.CRS("EPSG:4326")

            # Créer un transformateur entre les deux systèmes de coordonnées
            transformer = pyproj.Transformer.from_crs(epsg_2056, epsg_4326, always_xy=True)

            # Coordonnées des coins inférieur gauche (lower left) et supérieur droit (upper right) en EPSG:2056
            lower_left_x, lower_left_y = 2255000, 840000  # Remplacez par vos propres coordonnées
            upper_right_x, upper_right_y = 2965000, 1480000  # Remplacez par vos propres coordonnées

            # Convertir ces coordonnées en EPSG:4326 (longitude, latitude)
            lower_left_lon, lower_left_lat = transformer.transform(lower_left_x, lower_left_y)
            upper_right_lon, upper_right_lat = transformer.transform(upper_right_x, upper_right_y)


            # Création de la figure
            fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})  # Projection EPSG:2056

            # Affichage de la heatmap
            im = ax.imshow(df, extent=[lower_left_lon, upper_right_lon, lower_left_lat, upper_right_lat], origin='upper', cmap='viridis',
                        vmin=vmin, vmax=vmax)
            #ax.set_xlim(lon_min, lon_max)  # Nouvelles limites de longitude
            #ax.set_ylim(lat_min, lat_max)

            # Ajout d'une barre de couleur
            plt.colorbar(im, ax=ax, label="Precipitation during the passed hour (mm)")
            ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black')
            ax.add_feature(cfeature.COASTLINE, edgecolor='black')

            ax.gridlines(draw_labels=True, linestyle = ":", linewidth = .5)

            ax.set_title("Ground Truth from CPC" + "\n" + date)

            # Affichage
            pdf_fig.savefig()
            plt.close()
    


    def compress_pdf(input_pdf, output_pdf):
        # Assurez-vous que le chemin vers Ghostscript est correct
        gs_path = "gswin64c"  # Remplacez par "gswin32c" si vous êtes sur une machine 32-bit

        # Commande Ghostscript pour une compression d'image plus agressive
        gs_command = [
            gs_path,  # Ghostscript executable
            "-sDEVICE=pdfwrite",  # Format de sortie (PDF)
            "-dCompatibilityLevel=1.4",  # Version du PDF
            "-dNOPAUSE",  # Ne pas interrompre après chaque page
            "-dQUIET",  # Désactive les messages d'info
            "-dBATCH",  # Ferme Ghostscript une fois terminé
            "-dDownsampleColorImages=true",  # Active la réduction des images couleur
            "-dColorImageResolution=72",  # Réduit les images couleur à 72 dpi (ou 50 dpi pour plus de compression)
            "-dDownsampleGrayImages=true",  # Active la réduction pour les images en niveau de gris
            "-dGrayImageResolution=72",  # Réduit les images en niveaux de gris à 72 dpi
            "-dDownsampleMonoImages=true",  # Active la réduction pour les images monochromes
            "-dMonoImageResolution=72",  # Réduit les images monochromes à 72 dpi
            "-dJPEGQ=75",  # Compression des images couleur avec une qualité JPEG de 75 (à ajuster selon le compromis)
            "-dAutoFilterColorImages=true",  # Utilise une meilleure méthode de filtrage pour les images couleur
            "-sOutputFile=" + output_pdf,  # Spécifie le fichier de sortie
            input_pdf  # Fichier PDF d'entrée
        ]

        subprocess.run(gs_command, check=True)

    # Utilisation
    compress_pdf(f"Images/{ref_fichier}/figures.pdf", f"Images/{ref_fichier}/figures compressées.pdf")



         
    ########## Calcul des métriques ###################################################################




    def métrique(pred_ini, target, nom):
        with PdfPages(f"Images/{ref_fichier}/metrics {nom}.pdf") as pdf:

            target_array = np.asarray(target)

            ### Calcul du RMSE
            res = np.nanmean((pred_ini - target_array) ** 2) ** 0.5

            pdf_str = f"Root Mean Squared Error (RMSE): {str(res):.5}"


            ### Plot de la différence (au carré) des deux matrices
            diff_matrix = (pred_ini - target_array) 

            fig, ax = plt.subplots(figsize=(8, 10), subplot_kw={"projection": ccrs.PlateCarree()})

            # Ajout de la carte (côtes, frontières, etc.)
            ax.set_extent([lon_min_output, lon_max_output, lat_min_output, lat_max_output])
            ax.add_feature(cfeature.COASTLINE, linewidth=1)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle="dotted")

            # === 3. Affichage de la heatmap ===
            # On utilise imshow() avec l'étendue correcte des lat/lon
            img = ax.imshow(diff_matrix, extent=[lon_min_output, lon_max_output, lat_min_output, lat_max_output],
                            origin="upper", cmap="viridis", alpha=0.6)  # alpha pour transparence

            # === 4. Ajout d'une barre de couleur et finalisation ===
            plt.colorbar(img, orientation="vertical", label="Difference")
            ax.set_xticks([])
            ax.set_yticks([])
            plt.title(f"{nom} - target \n" + date)

            pdf.savefig()
            plt.close()


            ### Quantile plot
            # Aplatir les matrices
            pred = pred_ini.flatten()
            target_flat = target_array.flatten()

            nb_quantiles = 1000

            deciles1 = np.percentile(pred, np.linspace(1, 100, nb_quantiles))  # Déciles pour data1
            deciles2 = np.nanpercentile(target_flat, np.linspace(1, 100, nb_quantiles))  # Déciles pour data2

            # Créer un scatter plot pour comparer les déciles
            plt.figure(figsize=(8, 8))

            # Tracer les déciles de data1 contre les déciles de data2
            plt.scatter(deciles1, deciles2, color='b', label=f'Quantiles de {nom} vs target')

            # Ajouter des labels et une légende
            plt.xlabel(f'Quantiles de {nom}')
            plt.ylabel('Quantiles de Target')
            plt.title(f'QQ plot {nom} vs target')

            # Ajouter une ligne de référence à 45° pour montrer l'égalité des déciles
            plt.plot([min(deciles1), max(deciles1)], [min(deciles1), max(deciles1)], linestyle='--', color='black', label='Reference line')

            # Afficher la légende
            plt.legend()

            # Afficher le graphique
            plt.grid(True)
            pdf.savefig()
            plt.close()

            ######## Calcul de la statistique de Kolmogorov-Smirnov pour comparer les 2 distributions


            # Calcul de la distance KS entre data1 et data2
            statistic, p_value = ks_2samp(pred[~np.isnan(pred)], target_flat[~np.isnan(target_flat)])
            pdf_str += f"\nDistance Kolmogorov-Smirnov (KS): {str(statistic):.4}, p-value: {str(p_value):.4}"


            ######## Tracé de l'estimation de la densité de probabilité
            plt.figure(figsize=(10, 8))

            # Histograms pour les deux matrices
            plt.hist(pred, bins=200, density=True, label=f'{nom}', color='blue', histtype="step")
            plt.hist(target_flat, bins=200, density=True, label='Target', color='black', histtype="step")

            # Ajouter des labels et un titre
            plt.xlabel('Precipitation')
            plt.ylabel('Density')
            plt.yscale("log")
            plt.title(f'Approached distribution {nom} VS Target')
            plt.legend()

            # Afficher le graphique
            pdf.savefig()
            plt.close()


            ###### Calcul de l'erreur sur le 99.999 ème pourcentile
            p_true = np.nanpercentile(target_flat, 99.999)  # Percentile des vraies valeurs
            p_pred = np.nanpercentile(pred, 99.999)  # Percentile des prédictions
            error_99 = abs(p_true - p_pred)
            pdf_str += f"\n99.999th Percentile Error (PE) : {str(error_99):.4} mm"


            ##### Calculer la Earth-Mover Distance (Wasserstein Distance)
            emd = wasserstein_distance(pred[~np.isnan(pred)], target_flat[~np.isnan(target_flat)])

            pdf_str += f"\nEarth-Mover Distance (EMD) : {emd:.4}"

            ##### Calcul de la Spatial-Autocorrelation Error (SAE)



            # 1. Calcul des résidus (erreurs) en excluant les NaN dans la matrice de prédiction
            residuals = target_array - pred_ini

            # On remplit résidus de 0 pour les NaN, comme ça ça n'influence pas le calcul du I de Moran
            residuals = np.nan_to_num(residuals, 0)

            # 2. Création d'une matrice de voisins pour une grille 3x3
            # Dans ce cas, nous allons créer les relations de voisinage avec la topologie "Queen" 
            # qui connecte chaque cellule à ses voisins immédiats (horizontalement, verticalement et diagonalement)
            rows, cols = pred_ini.shape


            # Dictionnaires pour les voisins et les poids
            neighbors = {}
            weights = {}

            # Remplir les dictionnaires avec les voisins et les poids
            for i in range(rows):
                for j in range(cols):
                    current_cell = i * cols + j  # Indice linéaire pour la cellule (i, j)
                    
                    # Initialiser la liste des voisins et le dictionnaire des poids
                    neighbors[current_cell] = []
                    weights[current_cell] = []

                    # Vérifier les voisins (haut, bas, gauche, droite)
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:  # Vérifier si le voisin est dans la matrice
                            neighbor_cell = ni * cols + nj  # Indice linéaire pour le voisin
                            neighbors[current_cell].append(neighbor_cell)  # Ajouter le voisin
                            weights[current_cell].append(1)  # Poids de 1 pour chaque voisin

            # 3. Créer un objet de poids spatiaux W à partir des dictionnaires de voisins et de poids
            w = W(neighbors, weights)



            # 4. Calcul du coefficient de Moran sur les résidus
            moran = esda.Moran(residuals, w)




            pdf_str += f"\nSpatial Auto-Correlation Error (SAE): {str(moran.I):.4}"


            ##### Calcul du CRPS


            # Estimation de la densité de probabilité (KDE) pour les prédictions
            kde_pred = gaussian_kde(pred[np.isnan(pred) == False])
            # Estimation de la densité de probabilité (KDE) pour les cibles
            kde_target = gaussian_kde(target_flat[np.isnan(target_flat) == False])

            # Créer un ensemble de points pour évaluer les densités
            x_values = np.linspace(min(pred.min(), target_flat.min()) - 1, max(pred.max(), target_flat.max()) + 1, 1000)

            # Évaluer les densités pour chaque x
            density_pred = kde_pred(x_values)
            density_target = kde_target(x_values)

            # Fonction pour calculer le CRPS en utilisant l'approximation discrète
            def crps(density_pred, density_target):
                # On calcule la somme de la différence au carré entre les densités estimées
                return np.sum((density_pred - density_target) ** 2) / len(density_pred)

            # Calcul du CRPS
            crps_value = crps(density_pred, density_target)
            pdf_str += f"\nContinuous Ranked Probability Score (CRPS) : {crps_value:.4f}"


            # Écrire sur le pdf
            plt.figure(figsize=(8, 8))  # Format A4 en pouces

            plt.text(0, 0.5,  pdf_str, fontsize=12, verticalalignment="center", family="monospace", horizontalalignment='left')

            plt.axis("off")
            pdf.savefig()
            plt.close()

    métrique(ds_interp_iso, df, "Interpolation bicubique")
    métrique(ds_knn_iso, df, "kNN")


# On met des arguments en valeur qu'on l'exécute depuis le notebook
valeur = sys.argv[1]  # La valeur
pickle_file = sys.argv[2]  # Le chemin du fichier pickle contenant la ligne extraite

# Charger la ligne extraite depuis le fichier pickle
with open(pickle_file, 'rb') as f:
    ligne_extraite = pickle.load(f)

# Appeler la fonction principale
main(valeur, ligne_extraite)