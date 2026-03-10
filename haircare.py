import pandas as pd

haircare = pd.read_csv("datasets/haircare_cleaned.csv")


def get_haircare_user_input():
    print("\n=== Haircare Recommendation ===")
    hair_type= input("Hair Type (Curly/Straight/Wavy/Scalp issues/Damaged): ").strip().capitalize()
    routine_type = input("Routine Type (Wash/Condition/Treat/Style): ").strip().capitalize()
    return {"hair_type": hair_type, "routine_type": routine_type}

def recommend_haircare(user, top_n=5):
    df = haircare.copy()
    hair_match = df[df["hair_type"].str.contains(user["hair_type"], case=False, na=False)]

    if "routine_type" in df.columns:
        routine_match = hair_match[hair_match["routine_type"].str.contains(user["routine_type"], case=False, na=False)]
        if len(routine_match) >= top_n:
            results = routine_match
        elif len(hair_match) >= 1:
            results = hair_match
        else:
            results = df
    else:
        results = hair_match if len(hair_match) >= 1 else df

    top = results.head(top_n)
    print(f"\n Top {top_n} Haircare Products for '{user['hair_type']}' hair ({user['routine_type']} routine):")
    print("-" * 40)
    for i, (_, row) in enumerate(top.iterrows(), 1):
        print(f"  {i}. {row.get('product_name', 'N/A')}  |  Type: {row.get('hair_type', 'N/A')}  |  Routine: {row.get('routine_type', 'N/A')}")