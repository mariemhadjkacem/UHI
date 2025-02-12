# =======================================================
# Step 1: Import Libraries
# =======================================================
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import geopy.distance
from skopt import BayesSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin

# =======================================================
# Step 2: Load Datasets
# =======================================================
weather_data = pd.read_excel(r"data\NY_Mesonet_Weather.xlsx")
uhi_data = pd.read_csv(r"data\Training_data_uhi_index 2025-02-04.csv")

# =======================================================
# Step 3: Data Preprocessing
# =======================================================
weather_data['Date / Time'] = pd.to_datetime(weather_data['Date / Time'])
uhi_data['datetime'] = pd.to_datetime(uhi_data['datetime'])

# Merge datasets based on datetime
merged_data = pd.merge(uhi_data, weather_data, left_on='datetime', right_on='Date / Time', how='inner')

# Add additional time-based features
merged_data['month'] = merged_data['datetime'].dt.month
merged_data['day'] = merged_data['datetime'].dt.day
merged_data['weekofyear'] = merged_data['datetime'].dt.isocalendar().week
merged_data['hour'] = merged_data['datetime'].dt.hour

# Add lag features for previous temperature
merged_data['lag_temp'] = merged_data['Air Temp at Surface [degC]'].shift(1)

# Calculate weather trends: Rate of change for temperature, humidity, and wind speed
merged_data['temp_change'] = merged_data['Air Temp at Surface [degC]'].diff()
merged_data['humidity_change'] = merged_data['Relative Humidity [percent]'].diff()
merged_data['wind_speed_change'] = merged_data['Avg Wind Speed [m/s]'].diff()

# Add interaction features
merged_data['wind_solar'] = merged_data['Avg Wind Speed [m/s]'] * merged_data['Solar Flux [W/m^2]']
merged_data['humidity_temp'] = merged_data['Relative Humidity [percent]'] * merged_data['Air Temp at Surface [degC]']

# Feature Engineering: Geospatial features (distance to water body)
def get_distance_to_water(lat, lon):
    # Example coordinates for a water body (e.g., Hudson River)
    water_coords = (40.7128, -74.0060)
    return geopy.distance.distance((lat, lon), water_coords).km

merged_data['distance_to_water'] = merged_data.apply(lambda row: get_distance_to_water(row['Latitude'], row['Longitude']), axis=1)

# Add weather event categorization
def categorize_weather(row):
    if row['Air Temp at Surface [degC]'] < 0:
        return 'Snow'
    elif row['Solar Flux [W/m^2]'] > 500:
        return 'Sunny'
    elif row['Relative Humidity [percent]'] > 80:
        return 'Rain'
    else:
        return 'Cloudy'

merged_data['weather_event'] = merged_data.apply(categorize_weather, axis=1)

# Drop missing values for target variable
merged_data.dropna(subset=['Air Temp at Surface [degC]'], inplace=True)

# =======================================================
# Step 4: Define Features and Target
# =======================================================
X = merged_data[['Longitude', 'Latitude', 'month', 'day', 'weekofyear', 'hour', 'temp_change', 'humidity_change',
                 'wind_speed_change', 'Air Temp at Surface [degC]', 'Relative Humidity [percent]', 'Avg Wind Speed [m/s]',
                 'Wind Direction [degrees]', 'Solar Flux [W/m^2]', 'UHI_index', 'lag_temp', 'wind_solar',
                 'humidity_temp', 'distance_to_water']]

# Encoding categorical variable 'weather_event'
X = pd.get_dummies(X, columns=['weather_event'], drop_first=True)

y = merged_data['Air Temp at Surface [degC]']

# =======================================================
# Step 5: Preprocessing Pipeline
# =======================================================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X.columns),
    ])

# =======================================================
# Step 6: Model Initialization (XGBoost, LightGBM, Random Forest)
# =======================================================
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', 
                             colsample_bytree=0.3, learning_rate=0.1, 
                             max_depth=5, alpha=10, n_estimators=100)

lgb_model = lgb.LGBMRegressor(objective='regression', learning_rate=0.05, n_estimators=100)

rf_model = RandomForestRegressor(n_estimators=100)

# Ensemble model (Stacking) 
class StackedModel(BaseEstimator, RegressorMixin):
    def __init__(self, base_models):
        self.base_models = base_models
        
    def fit(self, X, y):
        self.models_ = [model.fit(X, y) for model in self.base_models]
        return self
    
    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1)

ensemble_model = StackedModel(base_models=[xgb_model, lgb_model, rf_model])

# =======================================================
# Step 7: Hyperparameter Tuning with Bayesian Optimization
# =======================================================
param_space = {
    'max_depth': (3, 10),
    'learning_rate': (0.01, 0.1, 'uniform'),
    'n_estimators': (50, 500),
    'subsample': (0.6, 1.0, 'uniform'),
    'colsample_bytree': (0.3, 0.7, 'uniform'),
    'min_child_weight': (1, 5)
}

opt = BayesSearchCV(ensemble_model, param_space, n_iter=20, cv=TimeSeriesSplit(n_splits=5), 
                    scoring='neg_mean_squared_error', random_state=42)
opt.fit(X, y)

# Best parameters from Bayesian optimization
print(f"Best parameters: {opt.best_params_}")

# Train model with the best parameters
best_model = opt.best_estimator_

# =======================================================
# Step 8: Evaluate Model
# =======================================================
y_pred = best_model.predict(X)
mae = mean_absolute_error(y, y_pred)
rmse = mean_squared_error(y, y_pred, squared=False)
r2 = r2_score(y, y_pred)
mape = mean_absolute_percentage_error(y, y_pred)

# Print evaluation metrics
print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R^2: {r2}')
print(f'MAPE: {mape}')

# =======================================================
# Step 9: Visualizations
# =======================================================
plt.scatter(y, y_pred)
plt.xlabel('Actual Temperature')
plt.ylabel('Predicted Temperature')
plt.title('Actual vs Predicted Temperature')
plt.show()

# =======================================================
# Step 10: Save the Model (Optional)
# =======================================================
import joblib
joblib.dump(best_model, 'xgboost_temperature_model_optimized_final.pkl')