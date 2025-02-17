import rasterio
import pandas as pd
import numpy as np

# Ouvrir le fichier NDVI.tiff
with rasterio.open("NDVI.tiff") as src:
    ndvi_array = src.read(1)  # Lire la première bande (NDVI)
    transform = src.transform  # Transformation affine pour conversion en coordonnées
    height, width = ndvi_array.shape  # Dimensions de l'image

    # Extraire les valeurs de NDVI et les stocker dans un DataFrame
    data = []
    for i in range(height):
        for j in range(width):
            lon, lat = transform * (j, i)  # Convertir en coordonnées géographiques
            ndvi_value = ndvi_array[i, j]
            if not np.isnan(ndvi_value):  # Filtrer les valeurs invalides
                data.append([lon, lat, ndvi_value])

# Convertir en DataFrame et enregistrer en CSV
df = pd.DataFrame(data, columns=["Longitude", "Latitude", "NDVI"])
df.to_csv("NDVI_data.csv", index=False)

print("✅ Extraction terminée ! Fichier NDVI_data.csv enregistré.")
