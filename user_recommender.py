import pandas as pd
import joblib

model = joblib.load("models/recommendation_model.pkl")
outfits = pd.read_csv("datasets/final_feature_engineered_dataset.csv")

def get_outfit_user_input():
    print("\n=== Outfit Recommendation (ML-Powered) ===")
    age= float(input("Age: "))
    height= float(input("Height (cm): "))
    weight= float(input("Weight (kg): "))
    gender= input("Gender (Male/Female): ").strip().capitalize()
    body_type= input("Body Type (Apple/Athletic/Hourglass/Pear/Rectangle): ").strip().capitalize()
    skin_tone= input("Skin Tone (Dark/Fair/Medium): ").strip().capitalize()
    hair_type= input("Hair Type (Curly/Straight/Wavy): ").strip().capitalize()
    hair_color= input("Hair Color (Black/Blonde/Brown): ").strip().capitalize()
    event= input("Event Type (College/Festival/Formal/Party/Reception/Wedding): ").strip().capitalize()
    return {
        "age": age, "height_cm": height, "weight_kg": weight,
        "gender": gender, "body_type": body_type, "skin_tone": skin_tone,
        "hair_type": hair_type, "hair_color": hair_color, "event": event
    }

def build_feature_row(user, outfit_row):
    bmi = user["weight_kg"] / ((user["height_cm"] / 100) ** 2)
    return {
        "age": user["age"], "height_cm": user["height_cm"],
        "weight_kg": user["weight_kg"], "BMI": bmi,
        "gender_Female":user["gender"] == "Female",
        "body_type_Apple":user["body_type"] == "Apple",
        "body_type_Athletic":user["body_type"] == "Athletic",
        "body_type_Hourglass":user["body_type"] == "Hourglass",
        "body_type_Pear":user["body_type"] == "Pear",
        "body_type_Rectangle":user["body_type"] == "Rectangle",
        "skin_tone_Dark":user["skin_tone"] == "Dark",
        "skin_tone_Fair":user["skin_tone"] == "Fair",
        "skin_tone_Medium":user["skin_tone"] == "Medium",
        "hair_type_Curly":user["hair_type"] == "Curly",
        "hair_type_Straight":user["hair_type"] == "Straight",
        "hair_type_Wavy":user["hair_type"] == "Wavy",
        "hair_color_Black":user["hair_color"] == "Black",
        "hair_color_Blonde":user["hair_color"] == "Blonde",
        "hair_color_Brown":user["hair_color"] == "Brown",
        "body_match_score":outfit_row["body_match_score"],
        "style_score":outfit_row["style_score"],
        "category_Gown":outfit_row["category_Gown"],
        "category_Kurti":outfit_row["category_Kurti"],
        "category_Lehenga":outfit_row["category_Lehenga"],
        "category_Saree":outfit_row["category_Saree"],
        "category_Suit":outfit_row["category_Suit"],
        "fit_type_Loose":outfit_row["fit_type_Loose"],
        "fit_type_Regular":outfit_row["fit_type_Regular"],
        "fit_type_Slim":outfit_row["fit_type_Slim"],
        "primary_color_Black":outfit_row["primary_color_Black"],
        "primary_color_Blue":outfit_row["primary_color_Blue"],
        "primary_color_Green":outfit_row["primary_color_Green"],
        "primary_color_Pink":outfit_row["primary_color_Pink"],
        "primary_color_Red":outfit_row["primary_color_Red"],
        "fabric_Cotton":outfit_row["fabric_Cotton"],
        "fabric_Net":outfit_row["fabric_Net"],
        "fabric_Silk":outfit_row["fabric_Silk"],
        "fabric_Velvet":outfit_row["fabric_Velvet"],
        "event_type_College":outfit_row["event_type_College"],
        "event_type_Festival":outfit_row["event_type_Festival"],
        "event_type_Formal":outfit_row["event_type_Formal"],
        "event_type_Party":outfit_row["event_type_Party"],
        "event_type_Reception":outfit_row["event_type_Reception"],
        "event_type_Wedding":outfit_row["event_type_Wedding"],
    }

def recommend_outfits(user, top_n=5):
    unique_dresses = outfits.drop_duplicates(subset="dress_id")
    event_col = f"event_type_{user['event']}"
    if event_col in unique_dresses.columns:
        filtered = unique_dresses[unique_dresses[event_col] == True]
        if len(filtered) < top_n:
            filtered = unique_dresses
    else:
        filtered = unique_dresses

    feature_rows = [build_feature_row(user, row) for _, row in filtered.iterrows()]
    X_candidates = pd.DataFrame(feature_rows)
    X_candidates = X_candidates[model.feature_names_in_]

    filtered = filtered.copy()
    filtered["predicted_rating"] = model.predict(X_candidates)
    top = filtered.sort_values("predicted_rating", ascending=False).head(top_n)

    print(f"\n Top {top_n} Outfit Recommendations for '{user['event']}':")
    print("-" * 40)
    for i, (_, row) in enumerate(top.iterrows(), 1):
        print(f"  {i}. Dress ID: {row['dress_id']}  |  Predicted Rating: {row['predicted_rating']:.2f}/5")