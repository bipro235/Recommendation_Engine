import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import requests
import concurrent.futures

@st.cache_data
def load_data():
    pivot_table_movies = pd.read_csv("pt.csv")
    pivot_table_movies.set_index('original_title', inplace=True)

    similarity_scores = cosine_similarity(pivot_table_movies)
    similarity_score = np.load("similarity.npy")

    movies_new_dataset = pd.read_csv("filtered_dataset.csv")
    movies_new_dataset.drop(columns={'Unnamed: 0.1'}, axis=1, inplace=True)
    movies_new_dataset.rename(columns={'movie_id': 'movieId'}, inplace=True)
    return pivot_table_movies, similarity_scores, similarity_score, movies_new_dataset

pivot_table_movies, similarity_scores, similarity_score, movies_new_dataset = load_data()

#movies_new_dataset.set_index('Unnamed: 0', inplace = True)

@st.cache
def fetch_poster(movie_name):
    api_key = "c63fff9bb208b44f81ded1bcf68ce8b0"
    base_url = "https://api.themoviedb.org/3/search/movie"
    params = {
        'api_key': api_key,
        'query': movie_name
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    if response.status_code == 200 and data['results']:
        movie_id = data['results'][0]['id']
        poster_path = data['results'][0]['poster_path']
        poster_url = f"https://image.tmdb.org/t/p/w500/{poster_path}"
        return poster_url
    else:
        return None

def collaborative_recommend(movie_name, similarity_scores):
    index = np.where(pivot_table_movies.index == movie_name)[0][0]
    similar_movies = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:6]
    similar_movie_indices = [i[0] for i in similar_movies]
    similar_movie_titles = pivot_table_movies.index[similar_movie_indices]
    similarity_scores = [i[1] for i in similar_movies]

    # Create a DataFrame with "title" and "similarity_scores"
    recommendations_df = pd.DataFrame({'original_title': similar_movie_titles, 'score': similarity_scores})

    return recommendations_df

print(collaborative_recommend('Alien', similarity_scores))


def content_based_recommend(movie_name):
    movie_index = movies_new_dataset[movies_new_dataset['original_title'] == movie_name].index[0]
    distances = similarity_score[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommendations = []
    scores = []
    for i in movies_list:
        recommendation_movie = movies_new_dataset.iloc[i[0]].original_title
        recommendations.append(recommendation_movie)
        recommendation_score = i[1]
        scores.append(recommendation_score)
    recommendation_df = pd.DataFrame({'original_title': recommendations, 'score': scores})
    return recommendation_df

print(content_based_recommend('Iron Man'))

def weighted_hybrid_recommendation_engine(movie_name, collaborative_filtering_weights, content_based_weights):
    content_based_recommendations = content_based_recommend(movie_name)
    collaborative_filtering_recommendations = collaborative_recommend(movie_name, similarity_scores)

    # Multiply recommendation by weights
    content_based_recommendations['score'] *= content_based_weights
    collaborative_filtering_recommendations['score'] *= collaborative_filtering_weights

    # Merge recommendation weights
    combined_recommendations = pd.concat([content_based_recommendations, collaborative_filtering_recommendations])

    # Sort the recommendation by weights score
    combined_recommendations.sort_values(by='score', ascending=False, inplace=True)

    return combined_recommendations


collaborative_filtering_weights = 0.6
content_based_weights = 0.4

Hybrid_rec = weighted_hybrid_recommendation_engine('Cars 2', collaborative_filtering_weights, content_based_weights)
Hybrid_rec.reset_index()
print(Hybrid_rec)

st.title("MOVIE RECOMMENDATION SYSTEM")

selected_movie_name = st.selectbox("Select a movie", list(movies_new_dataset['original_title'].values))

if st.button('Recommend'):
    recommendations = weighted_hybrid_recommendation_engine(
        selected_movie_name, content_based_weights, collaborative_filtering_weights
    )
    recommendations.reset_index(drop=True, inplace=True)
    recommended_movies = recommendations['original_title'].tolist()

    # Fetch posters asynchronously
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit the poster fetching tasks
        poster_urls = list(executor.map(fetch_poster, recommended_movies))

    num_columns = 3  # Number of columns to display the recommendations
    num_rows = (len(recommended_movies) - 1) // num_columns + 1
    cols = st.columns(num_columns)

    for row in range(num_rows):
        for col in cols:
            with col:
                movie_index = row * num_columns + cols.index(col)
                if movie_index < len(recommended_movies):
                    st.header(recommended_movies[movie_index])
                    if poster_urls[movie_index]:
                        # Display the poster
                        st.image(poster_urls[movie_index])
                    else:
                        st.write("No poster available.")

