import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting":GradientBoostingRegressor(n_estimators=100, random_state=42),
    "KNN":                KNeighborsRegressor(n_neighbors=5),
}

all_results = []

def compare_and_save(csv_path, drop_cols, save_prefix, label, best_save_path):
   
    print(f"  {label}")
    
    df = pd.read_csv(csv_path)
    y  = df["final_ratings"]
    X  = df.drop(columns=drop_cols, errors="ignore")
    
    X = X.fillna(0)        
    y = y.fillna(y.mean()) 

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    best_model = None
    best_r2    = -999
    best_name  = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = round(mean_squared_error(y_test, y_pred), 3)
        mae  = round(mean_absolute_error(y_test, y_pred), 3)
        r2   = round(r2_score(y_test, y_pred), 3)

        print(f"  {name:25} | RMSE: {rmse:.3f} | MAE: {mae:.3f} | R2: {r2:.3f}")

        # Save every model separately
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, f"models/{save_prefix}_{safe_name}.pkl")

        # Track best
        if r2 > best_r2:
            best_r2    = r2
            best_model = model
            best_name  = name

        # Store for CSV report
        all_results.append({
            "Category":label,
            "Model":name,
            "RMSE": rmse,
            "MAE":mae,
            "R2": r2,
            "Best":""
        })

    # Mark best in results
    for row in all_results:
        if row["Category"] == label and row["Model"] == best_name:
            row["Best"] = "Best"

    # Save best model for the app
    joblib.dump(best_model, best_save_path)

    print(f"\nBest Model : {best_name}")
    print(f"Best R²: {best_r2:.3f}")
    print(f"All 5 models saved in models/")
    print(f"Best model saved as: {best_save_path}")


# OUTFIT
compare_and_save(
    csv_path= "datasets/final_feature_engineered_dataset.csv",
    drop_cols = ["user_id", "dress_id", "final_ratings"],
    save_prefix= "outfit",
    label= "OUTFIT",
    best_save_path = "models/recommendation_model.pkl"
)

# SKINCARE
compare_and_save(
    csv_path       = "datasets/skincare_feature_engineered.csv",
    drop_cols      = ["user_id", "product_id", "product_name", "final_ratings"],
    save_prefix    = "skincare",
    label          = "SKINCARE",
    best_save_path = "models/skincare_model.pkl"
)

#HAIRCARE
compare_and_save(
    csv_path       = "datasets/haircare_feature_engineered.csv",
    drop_cols      = ["user_id", "product_id", "product_name", "final_ratings"],
    save_prefix    = "haircare",
    label          = "HAIRCARE",
    best_save_path = "models/haircare_model.pkl"
)

#SAVE COMPARISON REPORT 
report = pd.DataFrame(all_results)
report.to_csv("reports/model_comparison_results.csv", index=False)
print("\n" + "="*60)
print("  FINAL COMPARISON REPORT")
print("="*60)
print(report.to_string(index=False))
print("\nReport saved to: reports/model_comparison_results.csv")
print("All done!")
