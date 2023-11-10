import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the data from a text file ('data.txt')
data = pd.read_csv('data.txt')

# Step 1: Handling Missing Values
data = data.dropna()

# Step 2: Remove Duplicates
data = data.drop_duplicates()

# Step 3: Convert 'Rating' to numeric, handling non-numeric values
data['Rating'] = pd.to_numeric(data['Rating'], errors='coerce')

# Step 4: Drop rows with NaN in the 'Rating' column
data = data.dropna(subset=['Rating'])

# Step 5: Convert 'Rating' to integer
data['Rating'] = data['Rating'].astype(int)

# Step 6: Create User-Item Matrix
user_item_matrix = data.pivot(index='User', columns='Movie', values='Rating')

# Step 7: Collaborative Filtering Recommendation System
def recommend_movies(user, user_item_matrix, similarity_threshold=0.2):
    # Calculate cosine similarity between the user and all other users
    similarity = cosine_similarity(user_item_matrix, user_item_matrix.loc[user].values.reshape(1, -1))

    # Get users with similarity above the threshold
    similar_users = user_item_matrix.index[similarity.flatten() > similarity_threshold]

    # Find movies that the user hasn't rated
    unrated_movies = user_item_matrix.columns[user_item_matrix.loc[user].isnull()]

    # Predict ratings for unrated movies based on similar users
    predicted_ratings = user_item_matrix.loc[similar_users, unrated_movies].mean()

    # Sort movies by predicted rating
    recommended_movies = predicted_ratings.sort_values(ascending=False)

    return recommended_movies

# Step 8: Fallback System for Popular Movies
def popular_movies_fallback(n=5):
    # Recommend n most popular movies
    popular_movies = data['Movie'].value_counts().index[:n]
    return popular_movies

# Step 9: Command Line Interface
def run_cli():
    print("Welcome to the Movie Recommendation System!")
    user_name = input("Please enter your name: ")

    try:
        recommendations = recommend_movies(user_name, user_item_matrix)
        print(f"\nHello, {user_name}! Here are your personalized movie recommendations:")
        print(recommendations.head())
    except KeyError:
        print(f"\nNo ratings available for {user_name}. Recommending popular movies instead.")
        recommendations = popular_movies_fallback()
        print(f"\nPopular Movies:\n{', '.join(recommendations)}")

    # Step 10: Save the cleaned data to 'clean_data.txt'
    data.to_csv('clean_data.txt', index=False)

if __name__ == "__main__":
    run_cli()

