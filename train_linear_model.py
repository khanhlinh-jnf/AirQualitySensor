import os
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

# Tạo thư mục lưu model nếu chưa có
os.makedirs("linear_models", exist_ok=True)

# Đọc dữ liệu
df_train = pd.read_csv("data/train.csv")

features = ['o3op1', 'o3op2', 'no2op1', 'no2op2']
targets = ['OZONE', 'NO2']

X_train = df_train[features]

# Danh sách mô hình
model_defs = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1, max_iter=5000),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
    "HuberRegressor": HuberRegressor(alpha=0.0001),
    "SVR": SVR(kernel="linear", C=1.0, epsilon=0.1)
}

# Train & đánh giá cho mỗi biến mục tiêu
for target in targets:
    print(f"\nTarget: {target}")
    y_train = df_train[target]
    best_mae = float('inf')
    best_model_name = ""
    rows = []

    # Tên hệ số theo từng target
    if target == "OZONE":
        coef_names = ["po3", "qo3", "ro3", "so3"]
        intercept_name = "to3"
    elif target == "NO2":
        coef_names = ["pno2", "qno2", "rno2", "sno2"]
        intercept_name = "tno2"

    for name, model in model_defs.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        mae = mean_absolute_error(y_train, y_pred)

        print(f"{name} → MAE = {mae:.4f}")

        # Lưu model
        model_filename = f"linear_models/{name}_{target.lower()}.pkl"
        joblib.dump(model, model_filename)

		# Lấy hệ số phù hợp với từng loại model
        if hasattr(model, "coef_"):
            coefs = model.coef_
            if len(coefs.shape) == 2:
                coefs = coefs[0]  # với SVR sẽ là (1, n_features), cần lấy [0]
        elif hasattr(model, "dual_coef_"):
            coefs = model.dual_coef_[0] if hasattr(model, "dual_coef_") else [float("nan")] * len(coef_names)
        else:
            coefs = [float("nan")] * len(coef_names)

        # Lấy intercept phù hợp
        if hasattr(model, "intercept_"):
            intercept = model.intercept_
            if isinstance(intercept, (list, tuple)) or hasattr(intercept, "__len__"):
                intercept = intercept[0]
        else:
            intercept = float("nan")

        # Tạo row dữ liệu với tên hệ số
        row = {
            "Model": name,
            "MAE": mae,
            **{coef_names[i]: coefs[i] if i < len(coef_names) else float("nan") for i in range(len(coef_names))},
            intercept_name: intercept
        }

        rows.append(row)

        if mae < best_mae:
            best_mae = mae
            best_model_name = name

    # Lưu file kết quả CSV
    result_df = pd.DataFrame(rows)
    result_df.to_csv(f"linear_models/results_{target.lower()}.csv", index=False)

    print(f"Best model for {target}: {best_model_name} with MAE = {best_mae:.4f}")
	

