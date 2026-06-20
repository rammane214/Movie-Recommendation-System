 
# IMPORT LIBRARIES
 
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



# PAGE CONFIG

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)



# TITLE UI

st.markdown(
    "<h1 style='text-align:center; color:#FF4B4B;'>🎬 Movie Recommendation System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>Machine Learning + Business Analytics Dashboard</p>",
    unsafe_allow_html=True
)

st.divider()



# LOAD DATA

movies = pd.read_csv("../data/movies.csv")
ratings = pd.read_csv("../data/ratings.csv")

movies["genres"] = movies["genres"].fillna("")

movie_stats = ratings.groupby("movieId").agg(
    average_rating=("rating", "mean"),
    rating_count=("rating", "count")
).reset_index()

movies = movies.merge(movie_stats, on="movieId", how="left")

movies["average_rating"] = movies["average_rating"].fillna(0)
movies["rating_count"] = movies["rating_count"].fillna(0)



# ML MODEL

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()



# RECOMMENDATION FUNCTION

def hybrid_recommend(movie_name):

    if movie_name not in indices:
        return None

    idx = indices[movie_name]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]

    movie_indices = [i[0] for i in sim_scores]

    recommendations = movies.iloc[movie_indices][
        ["title", "genres", "average_rating", "rating_count"]
    ]

    return recommendations.sort_values(
        by=["average_rating", "rating_count"],
        ascending=False
    ).head(5)



# SIDEBAR MENU

st.sidebar.title("📌 Navigation")

option = st.sidebar.selectbox(
    "Choose Page",
    ["🏠 Home", "📊 Business Analytics", "🔥 Trending Movies", "🎯 Recommendations", "ℹ️ About"]
)


# HOME
if option == "🏠 Home":
    st.header("Welcome 👋")

    st.write("""
    This is a Movie Recommendation System built using:
    - Machine Learning (TF-IDF + Cosine Similarity)
    - Python
    - Streamlit Dashboard
    """)

    st.success("Use sidebar to explore features")


# BUSINESS ANALYTICS
elif option == "📊 Business Analytics":

    st.header("📊 Business Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("🎬 Total Movies", len(movies))

    with col2:
        st.metric("⭐ Total Ratings", len(ratings))

    st.subheader("🎭 Top Genres Distribution")
    genre_count = movies["genres"].value_counts().head(10)
    st.bar_chart(genre_count)


# TRENDING MOVIES
elif option == "🔥 Trending Movies":

    st.header("🔥 Trending Movies")

    movie_data = pd.merge(movies, ratings, on="movieId")

    popular = movie_data.groupby("title")["rating"].count()
    popular = popular.sort_values(ascending=False).head(10)

    st.bar_chart(popular)


# RECOMMENDATIONS
elif option == "🎯 Recommendations":

    st.header("🎯 Get Movie Recommendations")

    movie_name = st.selectbox(
        "🎬 Select a Movie",
        movies["title"].sort_values()
    )

    if st.button("Recommend 🎬"):

        result = hybrid_recommend(movie_name)

        if result is None:
            st.error("Movie not found")
        else:
            st.success("Top Recommendations")

            for _, row in result.iterrows():
                st.markdown("----")
                st.subheader("🎬 " + row["title"])
                st.write("🎭 Genres:", row["genres"])
                st.write("⭐ Rating:", round(row["average_rating"], 2))
                st.write("👥 Votes:", int(row["rating_count"]))


# ABOUT
elif option == "ℹ️ About":

    st.header("ℹ️ About This Project")

    st.write("""
    This project demonstrates:

    ✔ Movie Recommendation System  
    ✔ Machine Learning (TF-IDF + Cosine Similarity)  
    ✔ Business Analytics Dashboard  
    ✔ Streamlit Web App  
    ✔ Beginner Friendly ML Project  
    """)

    st.info("Great project for resume / portfolio 🚀")