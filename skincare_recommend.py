import pandas as pd

skincare = pd.read_csv("datasets/skincare_cleaned.csv")

def get_skincare_user_input():
    print("\n=== Skincare Recommendation ===")
    skin_type= input("Skin Type (Oily/Dry/Normal/Sensitive/Combination): ").strip().capitalize()
    routine_type= input("Routine Type (Wash/Moisturize/Treat/Protect): ").strip().capitalize()
    return {"skin_type": skin_type, "routine_type": routine_type}

def recommend_skincare(user, top_n=5):
    df = skincare.copy()
    skin_match = df[df["skin_type"].str.contains(user["skin_type"], case=False, na=False)]

    if "routine_type" in df.columns:
        routine_match = skin_match[skin_match["routine_type"].str.contains(user["routine_type"], case=False, na=False)]
        if len(routine_match) >= top_n:
            results = routine_match
        elif len(skin_match) >= 1:
            results = skin_match
        else:
            results = df
    else:
        results = skin_match if len(skin_match) >= 1 else df

    top = results.head(top_n)
    print(f"\n Top {top_n} Skincare Products for '{user['skin_type']}' skin ({user['routine_type']} routine):")
    print("-" * 40)
    for i, (_, row) in enumerate(top.iterrows(), 1):
        print(f"  {i}. {row.get('product_name', 'N/A')}  |  Type: {row.get('skin_type', 'N/A')}  |  Routine: {row.get('routine_type', 'N/A')}")