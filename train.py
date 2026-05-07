"""
train.py — Train the Linear Regression model and save it as a .pkl file
Run this once before launching the app: python train.py
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── 1. Load Data ──────────────────────────────────────────────────────────────
DATA_PATH = "data/tm271_cold_chain_spoilage_risk_dataset.csv"

df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# ── 2. Clean Data ─────────────────────────────────────────────────────────────
df.drop_duplicates(inplace=True)
if 'Shipment_ID' in df.columns:
    df.drop(columns=['Shipment_ID'], inplace=True)

# ── 3. KNN Imputation ─────────────────────────────────────────────────────────
num_cols = df.select_dtypes(include='number').columns
knn = KNNImputer(n_neighbors=5)
df[num_cols] = knn.fit_transform(df[num_cols])
print("✅ Missing values imputed with KNN")

# ── 4. Feature Engineering ────────────────────────────────────────────────────
df['Temp_Abuse_Index']          = df['Avg_Storage_Temp_C'] * df['Temp_Excursion_Hours']
df['Cooling_Protection_Ratio']  = (df['Packaging_Quality_Score'] + df['Ice_Replacement_Count']) / df['Transit_Duration_Hours']
df['Load_Stress_Index']         = (df['Vehicle_Load_Pct'] * df['Transit_Duration_Hours']) / 100
df['Thermal_Exposure_Index']    = (df['Avg_Storage_Temp_C'] + df['Ambient_Temp_C']) * df['Temp_Excursion_Hours']
df['Logistics_Inefficiency_Index'] = df['Fuel_Use_Liters'] / df['Distance_KM']

# ── 5. Features & Target ──────────────────────────────────────────────────────
MODEL_FEATURES = [
    'Avg_Storage_Temp_C', 'Temp_Excursion_Hours', 'Relative_Humidity_Pct',
    'Transit_Duration_Hours', 'Distance_KM', 'Packaging_Quality_Score',
    'Vehicle_Load_Pct', 'Door_Open_Events', 'Ice_Replacement_Count',
    'Ambient_Temp_C', 'Inspection_Hygiene_Score', 'Fuel_Use_Liters',
    'Temp_Abuse_Index', 'Cooling_Protection_Ratio', 'Load_Stress_Index'
]

X = df[MODEL_FEATURES]
y = df['Spoilage_Risk_Score']

# ── 6. Split ──────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── 7. Scale ──────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 8. Train ──────────────────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ── 9. Evaluate ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"\n📊 Model Performance:")
print(f"   MAE  = {mae:.3f}")
print(f"   RMSE = {rmse:.3f}")
print(f"   R²   = {r2:.3f}")

# ── 10. Save Artifacts ────────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)

with open("model/linear_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model/features.pkl", "wb") as f:
    pickle.dump(MODEL_FEATURES, f)

print("\n✅ Saved: model/linear_model.pkl")
print("✅ Saved: model/scaler.pkl")
print("✅ Saved: model/features.pkl")
print("\n🚀 Ready! Run: streamlit run app.py")
