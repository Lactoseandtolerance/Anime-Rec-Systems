import pandas as pd

#hi

# Load the dataset
anime_data = pd.read_csv('/Users/professornirvar/Downloads/anime.csv')

# Preprocess the data
'''ask for '''
# Handle missing values, clean the dataset, etc.

# Create the user-item matrix
user_item_matrix = anime_data.pivot_table(index='user_id', columns='anime_id', values='rating')

# Calculate the cosine similarity matrix
item_similarity = cosine_similarity(user_item_matrix.fillna(0))

# Function to get top recommendations for a user
def get_recommendations(user_id, top_n=5):
    user_ratings = user_item_matrix.loc[user_id].fillna(0)
    sim_scores = pd.Series(item_similarity[user_ratings.index[-1]], index=user_item_matrix.columns)
    top_items = sim_scores.sort_values(ascending=False)[:top_n]
    return top_items

# Generate recommendations for a specific user
user_id = 123  # Specify the user ID
recommendations = get_recommendations(user_id, top_n=10)

# Print the recommended anime titles
recommended_anime = anime_data.loc[anime_data['anime_id'].isin(recommendations.index)]
print(recommended_anime[['anime_id', 'title']])
