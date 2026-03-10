
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


import os

# Create folder if it doesn't exist
if not os.path.exists("models"):
    os.makedirs("models")

# Load feature-engineered dataset

merged = pd.read_csv("datasets/final_feature_engineered_dataset.csv")  

# Separate features and target
y = merged['final_ratings']  # target variable
X = merged.drop(columns=['user_id', 'dress_id', 'final_ratings'])  # features (drop IDs and target)

#Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
 
# Train Random Forest Regressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


#Evaluate Model

y_pred = model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("=== Model Evaluation ===")
print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"R² Score: {r2:.3f}")


#Feature Importance
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 Important Features:")
print(importances.head(10))


#Save the trained model

joblib.dump(model, "models/recommendation_model.pkl")
print("\nTrained Random Forest model saved as 'recommendation_model.pkl'")