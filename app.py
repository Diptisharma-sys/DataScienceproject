from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

outfit_model   = joblib.load("models/recommendation_model.pkl")
skincare_model = joblib.load("models/skincare_model.pkl")
haircare_model = joblib.load("models/haircare_model.pkl")

outfits= pd.read_csv("datasets/final_feature_engineered_dataset.csv")
skincare_info= pd.read_csv("datasets/skincare_cleaned.csv")
haircare_info= pd.read_csv("datasets/haircare_cleaned.csv")
skincare_feat= pd.read_csv("datasets/skincare_feature_engineered.csv")
haircare_feat= pd.read_csv("datasets/haircare_feature_engineered.csv")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/outfit")
def outfit_page():
    return render_template("outfit.html")

@app.route("/skincare")
def skincare_page():
    return render_template("skincare.html")

@app.route("/haircare")
def haircare_page():
    return render_template("haircare.html")

@app.route("/recommend/outfit", methods=["POST"])
def recommend_outfit():
    data = request.json
    user = {
        "age": float(data["age"]), "height_cm": float(data["height_cm"]),
        "weight_kg": float(data["weight_kg"]), "gender": data["gender"],
        "body_type": data["body_type"], "skin_tone": data["skin_tone"],
        "hair_type": data["hair_type"], "hair_color": data["hair_color"],
        "event": data["event"],
    }
    unique_dresses = outfits.drop_duplicates(subset="dress_id")
    event_col = f"event_type_{user['event']}"
    filtered = unique_dresses[unique_dresses[event_col] == True] if event_col in unique_dresses.columns else unique_dresses
    if len(filtered) < 5:
        filtered = unique_dresses

    bmi = user["weight_kg"] / ((user["height_cm"] / 100) ** 2)

    def build_row(r):
        return {
            "age": user["age"], "height_cm": user["height_cm"], "weight_kg": user["weight_kg"], "BMI": bmi,
            "gender_Female": user["gender"] == "Female",
            "body_type_Apple": user["body_type"] == "Apple", "body_type_Athletic": user["body_type"] == "Athletic",
            "body_type_Hourglass": user["body_type"] == "Hourglass", "body_type_Pear": user["body_type"] == "Pear",
            "body_type_Rectangle": user["body_type"] == "Rectangle",
            "skin_tone_Dark": user["skin_tone"] == "Dark", "skin_tone_Fair": user["skin_tone"] == "Fair",
            "skin_tone_Medium": user["skin_tone"] == "Medium",
            "hair_type_Curly": user["hair_type"] == "Curly", "hair_type_Straight": user["hair_type"] == "Straight",
            "hair_type_Wavy": user["hair_type"] == "Wavy",
            "hair_color_Black": user["hair_color"] == "Black", "hair_color_Blonde": user["hair_color"] == "Blonde",
            "hair_color_Brown": user["hair_color"] == "Brown",
            "body_match_score": r["body_match_score"], "style_score": r["style_score"],
            "category_Gown": r["category_Gown"], "category_Kurti": r["category_Kurti"],
            "category_Lehenga": r["category_Lehenga"], "category_Saree": r["category_Saree"],
            "category_Suit": r["category_Suit"], "fit_type_Loose": r["fit_type_Loose"],
            "fit_type_Regular": r["fit_type_Regular"], "fit_type_Slim": r["fit_type_Slim"],
            "primary_color_Black": r["primary_color_Black"], "primary_color_Blue": r["primary_color_Blue"],
            "primary_color_Green": r["primary_color_Green"], "primary_color_Pink": r["primary_color_Pink"],
            "primary_color_Red": r["primary_color_Red"],
            "fabric_Cotton": r["fabric_Cotton"], "fabric_Net": r["fabric_Net"],
            "fabric_Silk": r["fabric_Silk"], "fabric_Velvet": r["fabric_Velvet"],
            "event_type_College": r["event_type_College"], "event_type_Festival": r["event_type_Festival"],
            "event_type_Formal": r["event_type_Formal"], "event_type_Party": r["event_type_Party"],
            "event_type_Reception": r["event_type_Reception"], "event_type_Wedding": r["event_type_Wedding"],
        }

    X = pd.DataFrame([build_row(r) for _, r in filtered.iterrows()])[outfit_model.feature_names_in_]
    filtered = filtered.copy()
    filtered["predicted_rating"] = outfit_model.predict(X)
    top5 = filtered.sort_values("predicted_rating", ascending=False).head(5)

    results = []
    for _, row in top5.iterrows():
        category = next((c.replace("category_","") for c in ["category_Gown","category_Kurti","category_Lehenga","category_Saree","category_Suit"] if row.get(c, False)), "Unknown")
        color    = next((c.replace("primary_color_","") for c in ["primary_color_Black","primary_color_Blue","primary_color_Green","primary_color_Pink","primary_color_Red"] if row.get(c, False)), "Unknown")
        fabric   = next((c.replace("fabric_","") for c in ["fabric_Cotton","fabric_Net","fabric_Silk","fabric_Velvet"] if row.get(c, False)), "Unknown")
        fit      = next((c.replace("fit_type_","") for c in ["fit_type_Loose","fit_type_Regular","fit_type_Slim"] if row.get(c, False)), "Unknown")
        results.append({"dress_id": row["dress_id"], "predicted_rating": round(float(row["predicted_rating"]), 2),
                        "category": category, "color": color, "fabric": fabric, "fit": fit})
    return jsonify({"results": results})

@app.route("/recommend/skincare", methods=["POST"])
def recommend_skincare():
    data = request.json
    user = {
        "skin_type":        data["skin_type"].strip().capitalize(),
        "product_category": data["product_category"].strip().capitalize(),
    }

    # Build one feature row
    base = {"skin_match_score": 1.0}
    feature_row = pd.DataFrame([base])
    for col in skincare_model.feature_names_in_:
        if col not in feature_row.columns:
            feature_row[col] = 0
    for key, prefix in [("skin_type", "skin_type"), ("product_category", "product_category")]:
        col = f"{prefix}_{user[key]}"
        if col in skincare_model.feature_names_in_:
            feature_row[col] = 1
    feature_row = feature_row[skincare_model.feature_names_in_]

    # Repeat same row for all products; predict ALL at once
    n = len(skincare_info)
    X_all = pd.concat([feature_row] * n, ignore_index=True)
    preds = skincare_model.predict(X_all)

    results = []
    for i, (_, prod) in enumerate(skincare_info.iterrows()):
        results.append({
            "product_id":       str(prod["product_id"]),
            "product_name":     prod["product_name"],
            "skin_type":        prod["skin_type"],
            "product_category": prod["product_category"],
            "price_range":      prod["price_range"],
            "predicted_rating": round(float(preds[i]), 2)
        })

    results = sorted(results, key=lambda x: x["predicted_rating"], reverse=True)[:5]
    return jsonify({"results": results})
     
@app.route("/recommend/haircare", methods=["POST"])
def recommend_haircare():
    data = request.json
    user = {
        "hair_type":    data["hair_type"].strip().capitalize(),
        "routine_type": data["routine_type"].strip().capitalize(),
    }

    base = {"hair_match_score": 1.0}
    feature_row = pd.DataFrame([base])
    for col in haircare_model.feature_names_in_:
        if col not in feature_row.columns:
            feature_row[col] = 0
    for key, prefix in [("hair_type", "user_hair_type"), ("routine_type", "routine_type")]:
        col = f"{prefix}_{user[key]}"
        if col in haircare_model.feature_names_in_:
            feature_row[col] = 1
    feature_row = feature_row[haircare_model.feature_names_in_]

    # Predict ALL at once
    n = len(haircare_info)
    X_all = pd.concat([feature_row] * n, ignore_index=True)
    preds = haircare_model.predict(X_all)

    results = []
    for i, (_, prod) in enumerate(haircare_info.iterrows()):
        results.append({
            "product_id":       str(prod["product_id"]),
            "product_name":     prod["product_name"],
            "hair_type":        prod["hair_type"],
            "routine_type":     prod["routine_type"],
            "predicted_rating": round(float(preds[i]), 2)
        })

    results = sorted(results, key=lambda x: x["predicted_rating"], reverse=True)[:5]
    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(debug=True)