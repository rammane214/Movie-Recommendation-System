# Movie Recommendation System

A Streamlit-based movie recommendation system built with Python and machine learning.

## Overview

This project recommends movies using a hybrid approach based on movie genre similarity and ratings data. It includes:

- A Streamlit dashboard for exploration and recommendations
- TF-IDF vectorization of movie genres
- Cosine similarity to find similar movies
- Business analytics views for ratings and trending movies

## Repository Structure

```
Movie-Recommendation-System/
├── app/
│   └── app.py            # Streamlit application
├── data/
│   ├── links.csv
│   ├── movies.csv
│   ├── ratings.csv
│   └── tags.csv
├── images/               # Project images and assets
├── notebooks/            # Exploratory notebooks
└── README.md             # Project documentation
```

## Requirements

Recommended Python packages:

- streamlit
- pandas
- scikit-learn

If a `requirements.txt` file is not present, install the packages manually:

```bash
pip install streamlit pandas scikit-learn
```

## Running the App

From the `Movie-Recommendation-System` folder, run:

```bash
streamlit run app/app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

## Features

- `🏠 Home` — project overview and instructions
- `📊 Business Analytics` — summary metrics, movie counts, and genre analytics
- `🔥 Trending Movies` — top movies by rating counts
- `🎯 Recommendations` — choose a movie and get top similar recommendations
- `ℹ️ About` — project details and use cases

## Data

The app loads data from `data/movies.csv` and `data/ratings.csv`.

- `movies.csv` contains movie titles, genres, and metadata
- `ratings.csv` contains user rating values for movies

## Notes

- The recommendation engine is based on genre similarity and rating aggregation
- This project is suitable for portfolio demonstrations and learning recommender systems
- You can extend it with user-based filtering, additional metadata, or an API backend

## License

This repository can be used for educational and demo purposes.
