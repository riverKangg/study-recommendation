import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 가상의 고객 데이터
customer_data = {
    'CustomerID': [1, 2, 3],
    'Action': [5, 2, 1],
    'Comedy': [1, 5, 3],
    'Drama': [3, 4, 2]
}
customer_df = pd.DataFrame(customer_data)
customer_df.set_index('CustomerID', inplace=True)

# 가상의 영화 데이터
movie_data = {
    'MovieID': [101, 102, 103],
    'Action': [5, 4, 2],
    'Comedy': [1, 2, 5],
    'Drama': [2, 3, 4]
}
movie_df = pd.DataFrame(movie_data)
movie_df.set_index('MovieID', inplace=True)

# 코사인 유사도 계산
cosine_sim = cosine_similarity(customer_df, movie_df)

# 유사도를 기반으로 추천 생성
def get_recommendations(customer_id):
    sim_scores = list(enumerate(cosine_sim[customer_id - 1]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores]
    recommended_movies = movie_df.iloc[movie_indices]
    return recommended_movies

# 예시: 고객 1에게 영화 추천
recommendations = get_recommendations(1)
print(recommendations)
