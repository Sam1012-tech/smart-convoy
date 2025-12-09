import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import joblib

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------
df = pd.read_csv("synthetic_trips.csv")

print("Detected Columns:", df.columns.tolist())

# Target column
df["travel_time_min"] = df["duration_s"] / 60   # create target

TARGET = "travel_time_min"

# Select numerical features
FEATURES = [
    "start_lat", "start_lon",
    "end_lat", "end_lon",
    "distance_m", "duration_s",
    "traffic_level"
]
print("Using Features:", FEATURES)

X = df[FEATURES].astype(float)
y = df[TARGET].astype(float)

# Convert to numpy (fixes the NumPy 2.0 'copy=False' issue)
X = np.asarray(X)
y = np.asarray(y)

# ---------------------------------------------------
# Split the data
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------
# Train LightGBM Model
# ---------------------------------------------------
model = lgb.LGBMRegressor(
    n_estimators=800,
    learning_rate=0.05,
    num_leaves=35,
    random_state=42
)

# FIX: LightGBM now uses callbacks for early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(40)]
)

# ---------------------------------------------------
# Evaluate Model
# ---------------------------------------------------
pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("\n--- MODEL PERFORMANCE ---")
print("RMSE:", rmse)
print("MAE:", mae)
print("R²:", r2)

# ---------------------------------------------------
# Save model
# ---------------------------------------------------
joblib.dump(model, "eta_model.pkl")
joblib.dump(FEATURES, "eta_features.pkl")

print("\nModel saved successfully as eta_model.pkl")
