import joblib
import pandas as pd
import numpy as np

model_name = ["LinearRegression", "Ridge", "Lasso", "ElasticNet", "HuberRegressor", "SVR", "RandomForest", "GradientBoosting", "ExtraTrees", "DecisionTree", "KNN", "MLP"]

print("Choose model: ")
for i in range(len(model_name)):
	print(f"{i}: {model_name[i]}")
  
choice = int(input("Enter your choice: "))

if choice < 0 or choice >= len(model_name):
	print("Invalid choice. Exiting.")
	exit()
 
if choice < 6:
	model_ozone = joblib.load(f"linear_models/{model_name[choice]}_ozone.pkl")
	model_no2 = joblib.load(f"linear_models/{model_name[choice]}_no2.pkl")

	print()
	o3op1 = float(input("o3op1: "))
	o3op2 = float(input("o3op2: "))
	no2op1 = float(input("no2op1: "))
	no2op2 = float(input("no2op2: "))
	
	new_data = pd.DataFrame([[o3op1, o3op2, no2op1, no2op2]], columns=['o3op1', 'o3op2', 'no2op1', 'no2op2'])

	predict_ozone = model_ozone.predict(new_data)
	predict_no2 = model_no2.predict(new_data)
	print(f"Predicted Ozone: {predict_ozone[0]:.4f}")
	print(f"Predicted NO2: {predict_no2[0]:.4f}")
 
else:
	model_ozone = joblib.load(f"advance_models/{model_name[choice]}_ozone.pkl")
	model_no2 = joblib.load(f"advance_models/{model_name[choice]}_no2.pkl")
	scaler = joblib.load("advance_models/scaler_task2.pkl")
	features = joblib.load("advance_models/features.pkl") 
	print()
	o3op1 = float(input("o3op1: "))
	o3op2 = float(input("o3op2: "))
	no2op1 = float(input("no2op1: "))
	no2op2 = float(input("no2op2: "))
	temp = float(input("temperature: "))
	humidity = float(input("humidity: "))
	time = input("time (YYYY-MM-DD HH:MM:SS): ")

	new_data = pd.DataFrame([[o3op1, o3op2, no2op1, no2op2, temp, humidity, time]],
		columns=['o3op1', 'o3op2', 'no2op1', 'no2op2', 'temp', 'humidity', 'Time']
	)

	new_data['Time'] = pd.to_datetime(new_data['Time'])
	new_data['hour'] = new_data['Time'].dt.hour
	new_data['sin_hour'] = np.sin(2 * np.pi * new_data['hour'] / 24)
	new_data['cos_hour'] = np.cos(2 * np.pi * new_data['hour'] / 24)

	X = new_data[features]
	X_scaled = scaler.transform(X)

	predict_ozone = model_ozone.predict(X_scaled)
	predict_no2 = model_no2.predict(X_scaled)

	print(f"\nPredict:")
	print(f"Predicted Ozone: {predict_ozone[0]:.4f}")
	print(f"Predicted NO2: {predict_no2[0]:.4f}")