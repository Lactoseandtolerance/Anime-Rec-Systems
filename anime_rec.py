import pandas as pd

def anime_recommender(csv_file):
    df = pd.read_csv(csv_file)

    user_input = input("Enter the name of an anime or a list of genres (comma-separated): ").strip()

    # Check if the user input is a list of genres
    if "," in user_input:
        user_genres = [genre.strip() for genre in user_input.split(",")]
        
        # Filter the DataFrame based on the provided genres
        filtered_df = df[df['Genres'].str.contains('|'.join(user_genres), case=False, regex=True)]
        
        # Sort the filtered DataFrame by rating in descending order
        recommended_anime = filtered_df.sort_values(by='Rating', ascending=False)
    else:
        # Filter the DataFrame based on the anime title
        recommended_anime = df[df['Title'].str.contains(user_input, case=False, regex=True)]

    # Display the recommended anime
    if not recommended_anime.empty:
        print("Recommended Anime:")
        print(recommended_anime[['Title', 'Genres', 'Rating']])
    else:
        print("No matching anime found.")

anime_recommender("anime_data.csv")
