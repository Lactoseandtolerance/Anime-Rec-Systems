import pandas as pd

def anime_recommender(csv_file):
    df = pd.read_csv(csv_file)

    user_input = input("Enter the name of an anime or a list of genres (comma-separated): ").strip()

    # Check if the user input is a list of genres
    if "," in user_input:
        user_genres = [genre.strip() for genre in user_input.split(",")]
        
        # Filter the DataFrame based on the provided genres
        filtered_df = df[df['genre'].str.contains(','.join(user_genres), case=False, regex=True)]
        
        # Sort the filtered DataFrame by rating in descending order
        recommended_anime = filtered_df.sort_values(by='rating', ascending=False)
    else:
        # Filter the DataFrame based on the anime title
        recommended_anime = df[df['name'].str.contains(user_input, case=False, regex=True)]

    if not recommended_anime.empty:
        print("Recommended Anime:")
        print(recommended_anime[['name', 'genre', 'rating']])
    else:
        print("No matching anime found.")

anime_recommender("/Users/professornirvar/Downloads/anime.csv")
