import rasterio
import pandas as pd
import numpy as np

# Ouvrir le fichier GeoTIFF
with rasterio.open("Landsat_LST.tiff") as src:
    lst_array = src.read(1)  # Lire la première bande (LST)
    transform = src.transform  # Transformation affine pour conversion en coordonnées
    height, width = lst_array.shape  # Dimensions de l'image

    # Extraire les valeurs de LST et convertir en dataframe
    data = []
    for i in range(height):
        for j in range(width):
            lon, lat = transform * (j, i)  # Convertir en coordonnées géographiques
            lst_value = lst_array[i, j]
            if lst_value > 0:  # Filtrer les valeurs invalides
                data.append([lon, lat, lst_value])

# Convertir en DataFrame et enregistrer en CSV
df = pd.DataFrame(data, columns=["Longitude", "Latitude", "LST"])
df.to_csv("LST_data.csv", index=False)

print("✅ Extraction terminée ! Fichier LST_data.csv enregistré.")
