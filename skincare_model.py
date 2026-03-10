import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

os.makedirs("models", exist_ok=True)

# TRAIN SKINCARE MODEL
print("=" * 50)
print("  TRAINING SKINCARE MODEL")
print("=" * 50)

skincare = pd.read_csv("datasets/skincare_feature_engineered.csv")

y_sk = skincare["final_ratings"]
X_sk = skincare.drop(columns=["user_id", "product_id", "product_name", "final_ratings"], errors="ignore")

X_train_sk, X_test_sk, y_train_sk, y_test_sk = train_test_split(
    X_sk, y_sk, test_size=0.2, random_state=42
)

sk_model = RandomForestRegressor(n_estimators=100, random_state=42)
sk_model.fit(X_train_sk, y_train_sk)

y_pred_sk = sk_model.predict(X_test_sk)
print(f"RMSE : {mean_squared_error(y_test_sk, y_pred_sk):.3f}")
print(f"MAE  : {mean_absolute_error(y_test_sk, y_pred_sk):.3f}")
print(f"R²   : {r2_score(y_test_sk, y_pred_sk):.3f}")

joblib.dump(sk_model, "models/skincare_model.pkl")
print("Skincare model saved: models/skincare_model.pkl\n")

# TRAIN HAIRCARE MODEL
print("=" * 50)
print("  TRAINING HAIRCARE MODEL")
print("=" * 50)

haircare = pd.read_csv("datasets/haircare_feature_engineered.csv")

y_hc = haircare["final_ratings"]
X_hc = haircare.drop(columns=["user_id", "product_id", "product_name", "final_ratings"], errors="ignore")

X_train_hc, X_test_hc, y_train_hc, y_test_hc = train_test_split(
    X_hc, y_hc, test_size=0.2, random_state=42
)

hc_model = RandomForestRegressor(n_estimators=100, random_state=42)
hc_model.fit(X_train_hc, y_train_hc)

y_pred_hc = hc_model.predict(X_test_hc)
print(f"RMSE : {mean_squared_error(y_test_hc, y_pred_hc):.3f}")
print(f"MAE  : {mean_absolute_error(y_test_hc, y_pred_hc):.3f}")
print(f"R²   : {r2_score(y_test_hc, y_pred_hc):.3f}")

joblib.dump(hc_model, "models/haircare_model.pkl")
print("Haircare model saved: models/haircare_model.pkl")


