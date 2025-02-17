import pandas as pd

# ========================================================
# Charger les fichiers CSV
# ========================================================
lst_file = "LST_data.csv"
ndvi_file = "NDVI_data.csv"
fusiondata_file = "mergedui.csv"

# Charger les données
lst_data = pd.read_csv(lst_file)
ndvi_data = pd.read_csv(ndvi_file)
fusiondata = pd.read_csv(fusiondata_file)
# Arrondir les coordonnées à 4 décimales par exemple
lst_data['Latitude'] = lst_data['Latitude'].round(4)
lst_data['Longitude'] = lst_data['Longitude'].round(4)

ndvi_data['Latitude'] = ndvi_data['Latitude'].round(4)
ndvi_data['Longitude'] = ndvi_data['Longitude'].round(4)

fusiondata['Latitude'] = fusiondata['Latitude'].round(4)
fusiondata['Longitude'] = fusiondata['Longitude'].round(4)


# ========================================================
# Vérifier les colonnes communes
# ========================================================
print("Colonnes de lst_data:", lst_data.columns)
print("Colonnes de ndvi_data:", ndvi_data.columns)
print("Colonnes de fusiondata:", fusiondata.columns)

# ========================================================
# Fusionner lst_data avec fusiondata sur 'Date_Time', 'Latitude' et 'Longitude'
# ========================================================
fusiondata = pd.merge(fusiondata, lst_data, on=['Latitude', 'Longitude'], how='inner')

# ========================================================
# Fusionner ndvi_data avec la nouvelle fusiondata
# ========================================================
fusiondata = pd.merge(fusiondata, ndvi_data, on=['Latitude', 'Longitude'], how='inner')

# ========================================================
# Vérifier les résultats de la fusion
# ========================================================
print(fusiondata.head())

# ========================================================
# Sauvegarder le fichier fusionné
# ========================================================
fusiondata.to_csv("data\\fusion_avec_lst_ndvi.csv", index=False)

print("Fusion terminée. Le fichier fusionné est sauvegardé sous 'fusion_avec_lst_ndvi.csv'.")
