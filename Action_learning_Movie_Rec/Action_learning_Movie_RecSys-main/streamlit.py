import streamlit as st
import psycopg2
import hashlib
import numpy as np
import pandas as pd
import tensorflow as tf
from fuzzywuzzy import fuzz

local_url = "http://localhost:8501"
network_url = "http://192.168.0.50:8501"


# Load movie titles from CSV
df_title = pd.read_csv(r'C:\Users\DEEPAKRAJ\PycharmProjects\pythonProject2\archive\movie_titles.csv',
                       encoding='ISO-8859-1')








# Function to get related movie suggestions based on input movie name
def get_related_movie_suggestions(movie_name, df_title):
    movie_titles = df_title['Name']
    related_movies = movie_titles[movie_titles.apply(lambda x: fuzz.partial_ratio(movie_name, x) > 70)]
    return related_movies.tolist()

# Function to create a PostgreSQL connection
def create_connection():
    return psycopg2.connect(
        dbname="actionl",
        user="postgres",
        password="1234",
        host="localhost",
        port="5432"
    )

# Function to create a users table if it doesn't exist
def create_table(conn):
    create_query = """
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username TEXT NOT NULL,
        password TEXT NOT NULL
    );
    """
    with conn.cursor() as cursor:
        cursor.execute(create_query)
    conn.commit()

# Function to hash the password before storing it in the database
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to check if the entered password matches the hashed password in the database
def verify_password(stored_password, entered_password):
    return stored_password == hashlib.sha256(entered_password.encode()).hexdigest()

# Function to create a new user account
def create_user(username, password):
    conn = create_connection()
    with conn:
        with conn.cursor() as cursor:
            insert_query = "INSERT INTO users (username, password) VALUES (%s, %s)"
            hashed_password = hash_password(password)
            cursor.execute(insert_query, (username, hashed_password))
    conn.commit()

# Function to check if the username exists in the database
def username_exists(username):
    conn = create_connection()
    with conn:
        with conn.cursor() as cursor:
            query = "SELECT * FROM users WHERE username = %s"
            cursor.execute(query, (username,))
            return cursor.fetchone() is not None

# Function to get movie recommendations using collaborative filtering
def get_movie_recommendations(user_id, top_n, df_title, num_items):
    # Load the collaborative filtering model (if not already loaded)
    if 'collab_filter_model' not in st.session_state:
        st.session_state.collab_filter_model = tf.keras.models.load_model(
            r'C:\Users\DEEPAKRAJ\PycharmProjects\pythonProject2\backend\item-based-collaborative_filtering_model.h5')

    # Assuming you have a DataFrame 'df_title' containing movie information
    movie_titles = df_title['Name']

    # Call the collaborative filtering model to get movie recommendations
    user_movies = df[df['Cust_Id'] == user_id]['Movie_Id']
    unrated_movies = np.setdiff1d(np.arange(num_items), user_movies)
    predicted_ratings = st.session_state.collab_filter_model.predict(unrated_movies).flatten()
    top_movie_indices = np.argsort(predicted_ratings)[::-1][:top_n]

    recommendations = movie_titles.iloc[top_movie_indices]
    return recommendations

def main():
    st.title("Movie Recommendation System")

    # Create the database and table if they don't exist
    conn = create_connection()
    create_table(conn)

    # Check if the user is signed in
    if 'is_signed_in' not in st.session_state:
        st.session_state.is_signed_in = False

    # Check if the user details are stored in session_state
    if 'user' not in st.session_state:
        st.session_state.user = None

    # Navigation
    page = st.sidebar.selectbox("Select a page:", ["Sign Up", "Sign In", "Movie Recommendation"])

    if page == "Sign Up":
        st.header("Create an Account")
        new_username = st.text_input("Username")
        new_password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")

        if st.button("Sign Up"):
            if new_username and new_password and confirm_password:
                if new_password == confirm_password:
                    if not username_exists(new_username):
                        create_user(new_username, new_password)
                        st.success("Account created! You can now sign in.")
                    else:
                        st.error("Username already exists. Please choose a different username.")
                else:
                    st.error("Passwords do not match. Please try again.")
            else:
                st.warning("Please enter a username and password.")

    elif page == "Sign In":
        st.header("Sign In")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Sign In"):
            if username and password:
                conn = create_connection()
                with conn:
                    with conn.cursor() as cursor:
                        query = "SELECT * FROM users WHERE username = %s"
                        cursor.execute(query, (username,))
                        user = cursor.fetchone()
                        if user and verify_password(user[2], password):
                            st.session_state.is_signed_in = True
                            st.session_state.user = user
                            st.success(f"Welcome back, {username}!")
                            st.balloons()
                        else:
                            st.error("Invalid username or password. Please try again.")
            else:
                st.warning("Please enter a username and password.")


    elif page == "Movie Recommendation":
        if st.session_state.is_signed_in:
            st.header("Movie Recommendation")

            # Get the user input for the movie they are searching for
            movie_name = st.text_input("Enter Movie")

            # Get related movie suggestions based on the user's input
            related_movies = get_related_movie_suggestions(movie_name, df_title)

            # Display the movie suggestions in a selectbox as the user types
            selected_movie = st.selectbox("Select Movie Recommendation", related_movies)

            # If a suggestion is selected, update the movie_name input with the selected movie
            if selected_movie:
                movie_name = selected_movie

            if st.button("Search Movie Recommendation"):
                if st.session_state.user:
                    user_id = st.session_state.user['id']
                    top_n_recommendations = 5


                    # You can filter the DataFrame based on the selected movie name
                    filtered_df = df_title[df_title['Name'] == movie_name]

                    if not filtered_df.empty:
                        # Assuming you have a 'get_movie_id_from_name()' function to get movie ID from movie name
                        movie_id = get_movie_id_from_name(filtered_df.iloc[0]['Name'])
                        if movie_id is not None:
                            # Get movie recommendations based on the user's selected movie ID
                            movie_recommendations = get_movie_recommendations(user_id, top_n_recommendations, df_title,
                                                                              num_items)

                            st.header("Movie Recommendations")
                            st.write(movie_recommendations)
                        else:
                            st.warning("Movie not found in the dataset.")
                    else:
                        st.warning("Movie not found in the dataset.")
                else:
                    st.warning("You must sign in to access movie recommendations.")
        else:
            st.warning("You must sign in to access movie recommendations.")


# Add the movie collage image to the app
movie_collage_url = "https://i.redd.it/4fxxbm4opjd31.jpg"
st.image(movie_collage_url, caption="Movie Collage")


if __name__ == "__main__":
    main()