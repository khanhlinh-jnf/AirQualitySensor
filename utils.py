import os
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error
import numpy as np
df_test = pd.read_csv("data/dummy_test.csv")

model_dir = "linear_models"
df_test = pd.read_csv("data/dummy_test.csv")
targets = ["OZONE", "NO2"]


features = ['o3op1', 'o3op2', 'no2op1', 'no2op2']
X_test = df_test[features]
results = []

for target in targets:
    y_test = df_test[target]

    for file in os.listdir("linear_models"):
        if file.endswith(f"{target.lower()}.pkl"):
            model_path = os.path.join("linear_models", file)
            model_name = file.replace(f"_{target.lower()}.pkl", "")

            model = joblib.load(model_path)
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            result = {
                "Model": model_name,
                "Target": target,
                "MAE": mae
            }
            results.append(result)

result_df = pd.DataFrame(results)
result_df.to_csv("linear_models/results_on_test.csv", index=False)

if 'Time' in df_test.columns:
    df_test['Time'] = pd.to_datetime(df_test['Time'])
    df_test['hour'] = df_test['Time'].dt.hour
    df_test['sin_hour'] = np.sin(2 * np.pi * df_test['hour'] / 24)
    df_test['cos_hour'] = np.cos(2 * np.pi * df_test['hour'] / 24)

features = [col for col in df_test.columns if col not in ['OZONE', 'NO2', 'Time', 'hour']]
X_test = df_test[features]

scaler = joblib.load("advance_models/scaler_task2.pkl")
X_test_scaled = scaler.transform(X_test)
rows = []

for target in targets:
    y_test = df_test[target]

    for file in os.listdir("advance_models"):
        if file.endswith(f"{target.lower()}.pkl"):
            model_path = os.path.join("advance_models", file)
            model_name = file.replace(f"_{target.lower()}.pkl", "")

            model = joblib.load(model_path)
            y_pred = model.predict(X_test_scaled)

            mae = mean_absolute_error(y_test, y_pred)
            row = {
                "Model": model_name,
                "Target": target,
                "MAE": mae
            }
            rows.append(row)

result_df = pd.DataFrame(rows)
result_df = result_df.sort_values(by=["Target", "MAE"])
result_df.to_csv("advance_models/results_on_test.csv", index=False)

