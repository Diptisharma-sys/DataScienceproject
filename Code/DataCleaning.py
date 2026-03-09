import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

users = pd.read_csv("UserProfile.csv")
dresses = pd.read_csv("dresscatalog.csv")
ratings = pd.read_csv("recommendation.csv")

print("Users Dataset:")
print(users.head())
print("\nDress Catalog:")
print(dresses.head())
print("\nRatings Dataset:")
print(ratings.head())

users.dropna(inplace=True)
dresses.dropna(inplace=True)
ratings.dropna(inplace=True)

users.drop_duplicates(inplace=True)
dresses.drop_duplicates(inplace=True)
ratings.drop_duplicates(inplace=True)

users['body_type'].value_counts().plot(kind='bar', title='Body Type Distribution')
plt.show()

users['skin_tone'].value_counts().plot(kind='bar', title='Skin Tone Distribution')
plt.show()

dresses['category'].value_counts().plot(kind='bar', title='Dress Category Distribution')
plt.show()

plt.hist(ratings['rating'], bins=5, edgecolor='black')
plt.title('Ratings Distribution')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

users['height_m'] = users['height_cm'] / 100
users['BMI'] = users['weight_kg'] / (users['height_m'] ** 2)

le = LabelEncoder()

user_cols = ['gender', 'body_type', 'skin_tone', 'hair_type', 'hair_color']
for col in user_cols:
    users[col] = le.fit_transform(users[col])

dress_cols = ['category', 'fit_type', 'primary_color', 'fabric', 'event_type']
for col in dress_cols:
    dresses[col] = le.fit_transform(dresses[col])

data = ratings.merge(dresses, on='dress_id', how='left')
print("\nMerged Dataset:")
print(data.head())

features = dresses.drop('dress_id', axis=1)
similarity = cosine_similarity(features)

def recommend_similar(dress_id, top_n=3):
    try:
        idx = dresses[dresses['dress_id'] == dress_id].index[0]
        scores = list(enumerate(similarity[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        return list(dresses.iloc[[i[0] for i in scores]]['dress_id'])
    except:
        return []

matrix = ratings.pivot_table(index='user_id',
                             columns='dress_id',
                             values='rating').fillna(0)

user_sim = cosine_similarity(matrix)

def recommend_cf(user_id, top_n=3):
    if user_id not in matrix.index:
        return []
    user_index = matrix.index.get_loc(user_id)
    sim_scores = list(enumerate(user_sim[user_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:3]
    similar_users = [matrix.index[i[0]] for i in sim_scores]
    rec = ratings[ratings['user_id'].isin(similar_users)]
    rec = rec.sort_values('rating', ascending=False)
    return list(rec['dress_id'].head(top_n))

def hybrid_recommend(user_id, sample_dress):
    content = recommend_similar(sample_dress)
    collab = recommend_cf(user_id)
    final = list(set(content + collab))
    return final

print("\nContent-based Recommendation for D10:")
print(recommend_similar("D10"))

print("\nCollaborative Filtering Recommendation for U169:")
print(recommend_cf("U169"))

print("\nHybrid Recommendation for U169 with D10:")
print(hybrid_recommend("U169", "D10"))