import joblib
import pandas as pd

model_name = ["LinearRegression", "Ridge", "Lasso", "ElasticNet", "HuberRegressor", "SVR"]

print("Choose model: ")
for i in range(len(model_name)):
	print(f"{i}: {model_name[i]}")
  
choice = int(input("Enter your choice: "))

# Load scaler và model
model_ozone = joblib.load(f"linear_models/{model_name[choice]}_ozone.pkl")
model_no2 = joblib.load(f"linear_models/{model_name[choice]}_no2.pkl")

# Dữ liệu đầu vào mới
print()
o3op1 = float(input("o3op1: "))
o3op2 = float(input("o3op2: "))
no2op1 = float(input("no2op1: "))
no2op2 = float(input("no2op2: "))
 
new_data = pd.DataFrame([[o3op1, o3op2, no2op1, no2op2]], columns=['o3op1', 'o3op2', 'no2op1', 'no2op2'])

# Chuẩn hóa & dự đoán
predict_ozone = model_ozone.predict(new_data)
predict_no2 = model_no2.predict(new_data)
print(f"Predicted Ozone: {predict_ozone[0]:.4f}")
print(f"Predicted NO2: {predict_no2[0]:.4f}")
