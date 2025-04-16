import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

os.makedirs("models/decision_tree_models", exist_ok=True)
df_train = pd.read_csv("data/train.csv")

if 'Time' in df_train.columns:
	df_train['Time'] = pd.to_datetime(df_train['Time'])
	df_train['hour'] = df_train['Time'].dt.hour
	df_train['sin_hour'] = np.sin(2 * np.pi * df_train['hour'] / 24)
	df_train['cos_hour'] = np.cos(2 * np.pi * df_train['hour'] / 24)

features = [col for col in df_train.columns if col not in ['OZONE', 'NO2', 'Time', 'hour']]
joblib.dump(features, "models/decision_tree_models/features.pkl")
targets = ['OZONE', 'NO2']
X = df_train[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "models/decision_tree_models/scaler_task2.pkl")

model_defs = {
	"DecisionTree-maxdepth3": DecisionTreeRegressor(random_state=42, max_depth=3),
	"DecisionTree-maxdepth5": DecisionTreeRegressor(random_state=42, max_depth=5),
	"DecisionTree-maxdepth11": DecisionTreeRegressor(random_state=42, max_depth=11),
	"DecisionTree-maxdepth17": DecisionTreeRegressor(random_state=42, max_depth=17),
	"DecisionTree-maxdepthNone": DecisionTreeRegressor(random_state=42),
}
rows = []
for target in targets:
	print(f"\nTarget: {target}")
	y = df_train[target]
	best_mae = float('inf')

	for name, model in model_defs.items():
		model.fit(X_scaled, y)
		y_pred = model.predict(X_scaled)
		mae = mean_absolute_error(y, y_pred)
		if name != "DecisionTree-maxdepthNone":
			print(f"{name} - MAE = {mae:.4f}")
		else:
			print(f"DecisionTree - MAE = {mae:.4f} with max_depth {model.tree_.max_depth}")

		joblib.dump(model, f"models/decision_tree_models/{name}_{target.lower()}.pkl")
		rows.append({"model": name, "target": target, "mae": mae})

		if mae < best_mae:
			best_mae = mae

result_df = pd.DataFrame(rows)
result_df.to_csv("models/decision_tree_models/results_summary.csv", index=False)