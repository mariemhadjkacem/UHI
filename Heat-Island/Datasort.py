import pandas as pd

# Charger le dataset
df = pd.read_csv('data\\Training_data_uhi_index 2025-02-04.csv')

# Convertir la colonne 'datetime' en type datetime
df['datetime'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M')

# Filtrer les données de 15h00 à 15h59
df_filtered = df[df['datetime'].dt.hour == 15]

# Trier les données par datetime et longitude
df_sorted = df_filtered.sort_values(by=['datetime', 'Longitude'])

# Optionnel: sauvegarder le dataframe trié
df_sorted.to_csv('Training_data_uhi_index_sorted.csv', index=False)

# Afficher les premières lignes pour vérifier le résultat
print(df_sorted.head())
