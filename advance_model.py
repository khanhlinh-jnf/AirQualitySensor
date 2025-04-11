import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

os.makedirs("advance_models", exist_ok=True)
df_train = pd.read_csv("data/train.csv")


if 'Time' in df_train.columns:
    df_train['Time'] = pd.to_datetime(df_train['Time'])
    df_train['hour'] = df_train['Time'].dt.hour
    df_train['sin_hour'] = np.sin(2 * np.pi * df_train['hour'] / 24)
    df_train['cos_hour'] = np.cos(2 * np.pi * df_train['hour'] / 24)

features = [col for col in df_train.columns if col not in ['OZONE', 'NO2', 'Time', 'hour']]
joblib.dump(features, "advance_models/features.pkl")
targets = ['OZONE', 'NO2']
X = df_train[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "advance_models/scaler_task2.pkl")

model_defs = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=100, random_state=42),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "SVR": SVR(kernel='rbf', C=1.0, epsilon=0.1),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "MLP": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
}
rows = []
for target in targets:
    print(f"\nTarget: {target}")
    y = df_train[target]
    best_mae = float('inf')
    best_model_name = ""

    for name, model in model_defs.items():
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        mae = mean_absolute_error(y, y_pred)
        print(f"{name} - MAE = {mae:.4f}")

        joblib.dump(model, f"advance_models/{name}_{target.lower()}.pkl")
        rows.append({"model": name, "target": target, "mae": mae})

        if mae < best_mae:
            best_mae = mae
            best_model_name = name

    print(f"Best model for {target}: {best_model_name} with MAE = {best_mae:.4f}")

result_df = pd.DataFrame(rows)
result_df.to_csv("advance_models/results_summary.csv", index=False)