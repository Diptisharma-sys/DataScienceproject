from user_recommender import recommend_outfits, get_outfit_user_input
from skincare_recommend import recommend_skincare, get_skincare_user_input
from haircare import recommend_haircare, get_haircare_user_input

def main():
    print("=" * 50)
    print("   PERSONAL STYLE & CARE RECOMMENDATION SYSTEM")
    print("=" * 50)

    while True:
        print("\nWhat would you like recommendations for?")
        print("  1. Outfit")
        print("  2. Skincare")
        print("  3. Haircare")
        print("  4. Exit")

        choice = input("\nEnter choice (1/2/3/4): ").strip()

        if choice == "1":
            user = get_outfit_user_input()
            recommend_outfits(user)
        elif choice == "2":
            user = get_skincare_user_input()
            recommend_skincare(user)
        elif choice == "3":
            user = get_haircare_user_input()
            recommend_haircare(user)
        elif choice == "4":
            print("\nGoodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()