import os
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

def train_linear_models(df):
    os.makedirs("linear_models", exist_ok=True)
    df_train = df.copy()
    features = ['o3op1', 'o3op2', 'no2op1', 'no2op2']
    targets = ['OZONE', 'NO2']
    X_train = df_train[features]

    model_defs = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1, max_iter=5000),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
        "HuberRegressor": HuberRegressor(alpha=0.0001),
        "SVR": SVR(kernel="linear", C=1.0, epsilon=0.1)
    }
    rows = []
    for target in targets:
        print(f"\nTarget: {target}")
        y_train = df_train[target]
        best_mae = float('inf')

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
            print(f"{name} - MAE = {mae:.4f}")
            model_filename = f"models/linear_models/{name}_{target.lower()}.pkl"
            joblib.dump(model, model_filename)

            if hasattr(model, "coef_"):
                coefs = model.coef_
                if len(coefs.shape) == 2:
                    coefs = coefs[0]
            elif hasattr(model, "dual_coef_"):
                coefs = model.dual_coef_[0] if hasattr(model, "dual_coef_") else [float("nan")] * len(coef_names)
            else:
                coefs = [float("nan")] * len(coef_names)

            if hasattr(model, "intercept_"):
                intercept = model.intercept_
                if isinstance(intercept, (list, tuple)) or hasattr(intercept, "__len__"):
                    intercept = intercept[0]
            else:
                intercept = float("nan")
                
            row = {
                "Model": name,
                "MAE": mae,
                **{coef_names[i]: coefs[i] if i < len(coef_names) else float("nan") for i in range(len(coef_names))},
                intercept_name: intercept
            }
            rows.append(row)

            if mae < best_mae:
                best_mae = mae

        
    result_df = pd.DataFrame(rows)
    result_df.to_csv(f"models/linear_models/results_summary.csv", index=False)