import pandas as pd 
import geopy.distance
import os  
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# =======================================================
# Step 2: Load Datasets
# =======================================================
weather_file = r"data\Manhattan.xlsx"
uhi_file = r"D:\Emna-Mariem\data\Training_data_uhi_index 2025-02-04.csv"

# Vérification des fichiers
if not all(map(os.path.exists, [weather_file, uhi_file])):
    raise FileNotFoundError("Error: One or both data files are missing.")

# Chargement des données
weather_data = pd.read_excel(weather_file, engine='openpyxl')
uhi_data = pd.read_csv(uhi_file)

# =======================================================
# Step 3: Fix Column Names
# =======================================================
weather_data.columns = weather_data.columns.str.strip()
uhi_data.columns = uhi_data.columns.str.strip()

weather_data.rename(columns={'Date/Time': 'Date_Time'}, inplace=True)

# Vérification de l'existence de la colonne UHI Index
uhi_index_column = next((col for col in uhi_data.columns if "UHI" in col), None)
if not uhi_index_column:
    raise KeyError("Error: 'UHI Index' column is missing in the UHI data.")

# =======================================================
# Step 4: Fix Datetime Issues
# =======================================================
uhi_data.rename(columns={'datetime': 'Datetime'}, inplace=True)
uhi_data['Datetime'] = pd.to_datetime(uhi_data['Datetime'], errors='coerce')
weather_data['Date_Time'] = pd.to_datetime(weather_data['Date_Time'].str.replace(" EDT", "", regex=False), errors='coerce')

# Suppression des valeurs NaT
uhi_data.dropna(subset=['Datetime'], inplace=True)

# =======================================================
# Step 5: Merge Data
# =======================================================
fusiondata = pd.merge(weather_data, uhi_data, left_on='Date_Time', right_on='Datetime', how='inner')

if fusiondata.empty:
    raise ValueError("⚠️ Aucune donnée après la fusion. Vérifie les fichiers source.")

fusiondata.drop(columns=['Datetime'], inplace=True)
fusiondata.sort_values(by='Date_Time', inplace=True)

# Sauvegarde des données fusionnées
output_file = "merged_uhi.csv"
fusiondata.to_csv(output_file, index=False)
print(f"✅ Fichier fusionné enregistré ici: {output_file}")

# =======================================================
# Step 7: Define Features and Target
# =======================================================
feature_cols = [
    'Air Temp at Surface [degC]',
    'Relative Humidity [percent]', 'Avg Wind Speed [m/s]',
    'Wind Direction [degrees]', 'Solar Flux [W/m^2]',
    'distance_to_water', 'LST', 'NDVI'
]

feature_cols = [col for col in feature_cols if col in fusiondata.columns]
X = fusiondata[feature_cols]
y = fusiondata[uhi_index_column]

if X.empty or y.empty:
    raise ValueError("Error: No data available for training. Check merging and missing values.")

# =======================================================
# Step 8: Train-Test Split
# =======================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =======================================================
# Step 9: Train and Evaluate Model
# =======================================================
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)
y_pred_test = model.predict(X_test)

# Évaluation du modèle
metrics = {
    'MAE': mean_absolute_error(y_test, y_pred_test),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
    'R^2': r2_score(y_test, y_pred_test),
    'MAPE': mean_absolute_percentage_error(y_test, y_pred_test)
}

for metric, value in metrics.items():
    print(f'Test {metric}: {value:.4f}')

print("Features utilisées pour l'entraînement :", X.columns.tolist())

# =======================================================
# Step 11: Visualizations
# =======================================================
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_test, alpha=0.6, edgecolors='k')
plt.xlabel('Actual UHI Index')
plt.ylabel('Predicted UHI Index')
plt.title('Actual vs Predicted UHI Index')
plt.grid(True)
plt.savefig("prediction_plot.png")
plt.show()
