import streamlit as st
import requests
import tensorflow as tf
import numpy as np
import pandas as pd

df = pd.read_csv('/Users/loicsteve/Desktop/EPITA/Semester 3/Action learning/data/final_data_set.csv')

API_KEY = "e9c3346ca46a25cec9b7804aebfc98d4"
BASE_URL = 'https://api.themoviedb.org/3/search/movie'

def get_movie_image_url(title):
    params = {
        'api_key': API_KEY,
        'query': title,
    }

    response = requests.get(BASE_URL, params=params)
    data = response.json()

    if 'results' in data and data['results']:
        movie = data['results'][0]  # Assuming the first result is the best match
        if 'poster_path' in movie:
            image_url = f"https://image.tmdb.org/t/p/original{movie['poster_path']}"
            return image_url

    return None


def recommend_items(user_id, top_n):
    if user_id not in df['Cust_Id'].values:
        # User is new and not in the dataset, recommend top N most rated movies
        average_ratings = df.groupby('Movie_Id')['Rating'].mean()
        sorted_movies = average_ratings.sort_values(ascending=False)
        recommended_movies = sorted_movies.head(top_n)
        recommendations = df_title.loc[recommended_movies.index]
    else:
        # Get the movies rated by the user
        user_rated_movies = df.loc[df['Cust_Id'] == user_id, 'Movie_Id']

        # Predict ratings for all items
        all_items = np.arange(num_items)
        predicted_ratings = model.predict(all_items)

        # Remove movies that the user has already rated
        unrated_items = np.setdiff1d(all_items, user_rated_movies)

        # Sort unrated items based on predicted ratings in descending order
        sorted_indices = np.argsort(predicted_ratings[unrated_items].flatten())[::-1]

        # Get the top N recommended items
        top_items = unrated_items[sorted_indices[:top_n]]

        # Retrieve movie information for the top recommended items
        recommendations = df_title.loc[df_title.index.isin(top_items)]

    # Add movie images to recommendations
    movie_images = []
    for movie in recommendations['Name']:
        image_url = get_movie_image_url(movie)
        if image_url:
            movie_images.append(image_url)
        else:
            movie_images.append('Image Not Found')  # Add a placeholder if image not found

    recommendations['Image_URL'] = movie_images
    return recommendations


# Load your data and model here (replace with your data and model loading code)
df_title = pd.read_csv('/Users/loicsteve/Desktop/EPITA/Semester 3/Action learning/data/movie_titles.csv', encoding='ISO-8859-1', header=None, names=['Movie_Id', 'Year', 'Name'])
num_items = len(df_title)  # Calculate the total number of movies in the dataset

# Assuming you have your trained model here (replace with your model loading code)
model = tf.keras.models.load_model('/Users/loicsteve/Desktop/EPITA/Semester 3/Action learning/data/item-based-collaborative_filtering_model.h5')


def main():
    st.title("Movie Recommendation System")

    # Display the recommendation button
    user_id = st.number_input("Enter User ID")
    if st.button("Get Movie Recommendations"):

        top_n_recommendations = 10

        # Get movie recommendations using the recommend_items function
        movie_recommendations = recommend_items(user_id, top_n_recommendations)

        st.header("Movie Recommendations")
        for idx, movie_row in movie_recommendations.iterrows():
            st.write(movie_row['Name'])
            image_url = movie_row['Image_URL']
            if image_url and image_url != 'Image Not Found':
                st.image(image_url, caption=movie_row['Name'], use_column_width=True)
            else:
                st.write("Image Not Found")
            st.write("---")

if __name__ == "__main__":
    main()
