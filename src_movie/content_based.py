import random
import pandas as pd
import numpy as np

# Read Data
df1 = pd.read_csv('input/tmdb-movie-metadata/tmdb_5000_credits.csv')
df1.columns = ['id', 'tittle', 'cast', 'crew']
# 'movie_id', 'title', 'cast', 'crew'
df2 = pd.read_csv('input/tmdb-movie-metadata/tmdb_5000_movies.csv')
# 'budget', 'genres', 'homepage', 'id', 'keywords', 'original_language',
#        'original_title', 'overview', 'popularity', 'production_companies',
#        'production_countries', 'release_date', 'revenue', 'runtime',
#        'spoken_languages', 'status', 'tagline', 'title', 'vote_average',
#        'vote_count'
rating = pd.read_csv('input/tmdb-movie-metadata/ratings_small.csv')
# 'userId', 'movieId', 'rating', 'timestamp'


# Content Based Filtering

# user features x item features
# 1. make virtual user data
user_df = rating[['userId']].drop_duplicates()
user_df['gender'] = np.random.choice([0, 1], size=len(user_df))
user_df['age'] = np.random.randint(10, 71, size=len(user_df))
num_features = 10
for i in range(num_features):
    val1 = random.randint(0, 1000)
    val2 = random.randint(1001, 2000)
    user_df[f'feaure_{i}'] = np.random.randint(val1, val2, size=len(user_df))
user_df.set_index('userId', inplace=True)

# 2. make item data
item_df = df2.select_dtypes(include=[np.number])  # Selected only numeric columns
item_df.rename(columns={'id': 'movieId'}, inplace=True)
item_df.dropna(axis=0, inplace=True)
item_df.set_index('movieId', inplace=True)

print(user_df.shape, item_df.shape)

# 3. Align the dimensions
# PCA
from sklearn.decomposition import PCA

n_components = 6
pca_user = PCA(n_components=n_components)
pca_item = PCA(n_components=n_components)

user_df_pca = pca_user.fit_transform(user_df)
item_df_pca = pca_item.fit_transform(item_df)

# t-SNE
from sklearn.manifold import TSNE

n_components = 6
tsne_user = TSNE(n_components=n_components)
tsne_item = TSNE(n_components=n_components)

user_df_tsne = tsne_user.fit_transform(user_df)
item_df_tsne = tsne_item.fit_transform(item_df)

# 4. Calculate similarity
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(user_df_pca, item_df_pca)
print(cosine_sim)


# 5. Recommend items
def get_recommendations(user_id):
    sim_scores = list(enumerate(cosine_sim[user_id - 1]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    item_indices = [i[0] for i in sim_scores]
    recommended_items = item_df.iloc[item_indices]
    return recommended_items


recommendations = get_recommendations(2)
print(recommendations)

#  - overview
# https://www.kaggle.com/code/ibtesama/getting-started-with-a-movie-recommendation-system/notebook
df2 = df2.merge(df1, on='id')
df2['overview'].head(5)
# Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

# Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

# Replace NaN with an empty string
df2['overview'] = df2['overview'].fillna('')

# Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df2['overview'])

# Output the shape of tfidf_matrix
tfidf_matrix.shape

# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Construct a reverse map of indices and movie titles
indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the titlef
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df2['title'].iloc[movie_indices]


get_recommendations('The Dark Knight Rises')
