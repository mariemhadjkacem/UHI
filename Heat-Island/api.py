# Supprimer les avertissements
import warnings
warnings.filterwarnings('ignore')

# Importation des outils communs GIS
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import rioxarray as rio
import rasterio
from matplotlib.cm import jet, RdYlGn

# Importation des outils Planetary Computer
import stackstac
import pystac_client
import planetary_computer
from odc.stac import stac_load

# Définir les limites géographiques de la zone d'intérêt (New York City)
lower_left = (40.75, -74.01)
upper_right = (40.88, -73.86)
bounds = (lower_left[1], lower_left[0], upper_right[1], upper_right[0])

# Définir la période de recherche des données
time_window = "2021-06-01/2021-09-01"

# Initialiser le client STAC pour accéder à l'API Planetary Computer
stac = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

# Effectuer une recherche pour les données Landsat
search = stac.search(
    bbox=bounds, 
    datetime=time_window,
    collections=["landsat-c2-l2"],
    query={"eo:cloud_cover": {"lt": 50}, "platform": {"in": ["landsat-8"]}},
)

# Récupérer les éléments de la recherche
items = list(search.get_items())
print('This is the number of scenes that touch our region:', len(items))

# Signer les éléments pour les authentifier avec Planetary Computer
signed_items = [planetary_computer.sign(item).to_dict() for item in items]

# Définir la résolution du produit final en mètres par pixel
resolution = 30  # 30 mètres par pixel
scale = resolution / 111320.0  # Conversion en degrés pour le CRS 4326

# Charger les données pour les bandes RGB (Red, Green, Blue, NIR)
data1 = stac_load(
    items,
    bands=["red", "green", "blue", "nir08"],
    crs="EPSG:4326",
    resolution=scale,
    chunks={"x": 2048, "y": 2048},
    dtype="uint16",
    patch_url=planetary_computer.sign,
    bbox=bounds
)

# Charger les données pour la bande Surface Temperature (lwir11)
data2 = stac_load(
    items,
    bands=["lwir11"],
    crs="EPSG:4326",
    resolution=scale,
    chunks={"x": 2048, "y": 2048},
    dtype="uint16",
    patch_url=planetary_computer.sign,
    bbox=bounds
)

# Appliquer les facteurs de mise à l'échelle pour les bandes RGB et NIR
scale1 = 0.0000275
offset1 = -0.2
data1 = data1.astype(float) * scale1 + offset1

# Appliquer les facteurs de mise à l'échelle pour la bande Surface Temperature
scale2 = 0.00341802
offset2 = 149.0
kelvin_celsius = 273.15  # Conversion de Kelvin à Celsius
data2 = data2.astype(float) * scale2 + offset2 - kelvin_celsius

# Calculer l'Indice de Végétation (NDVI)
ndvi_data = (data1.isel(time=0).nir08 - data1.isel(time=0).red) / (data1.isel(time=0).nir08 + data1.isel(time=0).red)

# Afficher l'Indice de Végétation (NDVI)
fig, ax = plt.subplots(figsize=(11, 10))
ndvi_data.plot.imshow(vmin=0.0, vmax=1.0, cmap="RdYlGn")
plt.title("Vegetation Index = NDVI")
plt.axis('off')
plt.show()

# Afficher la Température de Surface (LST)
fig, ax = plt.subplots(figsize=(11, 10))
data2.isel(time=0).lwir11.plot.imshow(vmin=20.0, vmax=45.0, cmap="jet")
plt.title("Land Surface Temperature (LST)")
plt.axis('off')
plt.show()

# Sélectionner un seul slice temporel pour la sortie
data3 = data2.isel(time=0)
filename = "Landsat_LST.tiff"  # Nom du fichier de sortie

# Calculer les dimensions du fichier
height = data3.dims["latitude"]
width = data3.dims["longitude"]

# Définir le Système de Référence de Coordonnées (CRS) comme étant les coordonnées Lat-Lon
gt = rasterio.transform.from_bounds(lower_left[1], lower_left[0], upper_right[1], upper_right[0], width, height)

# Écrire le CRS et la transformation sur l'objet xarray
data3.rio.write_crs("epsg:4326", inplace=True)
data3.rio.write_transform(transform=gt, inplace=True)

# Créer le fichier GeoTIFF de sortie
with rasterio.open(filename, 'w', driver='GTiff', width=width, height=height,
                   crs='epsg:4326', transform=gt, count=1, compress='lzw', dtype='float64') as dst:
    dst.write(data3.lwir11, 1)
    dst.close()

# Afficher le fichier GeoTIFF nouvellement créé
import os

# Lister les fichiers .tiff dans le répertoire actuel
for file in os.listdir('.'):
    if file.endswith('.tiff'):
        print(file)




# Définir le nom du fichier de sortie
filename_ndvi = "NDVI.tiff"

# Calculer les dimensions du fichier
height_ndvi = ndvi_data.shape[0]
width_ndvi = ndvi_data.shape[1]

# Définir la transformation géographique
gt_ndvi = rasterio.transform.from_bounds(lower_left[1], lower_left[0], upper_right[1], upper_right[0], width_ndvi, height_ndvi)

# Écrire le NDVI dans un fichier GeoTIFF
with rasterio.open(filename_ndvi, 'w', driver='GTiff', width=width_ndvi, height=height_ndvi,
                   crs='epsg:4326', transform=gt_ndvi, count=1, compress='lzw', dtype='float64') as dst:
    dst.write(ndvi_data.values, 1)
    dst.close()

print(f"✅ Fichier NDVI.tiff enregistré avec succès !")

