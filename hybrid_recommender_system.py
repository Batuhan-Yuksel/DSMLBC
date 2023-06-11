##################################
# # # # # BATUHAN YÜKSEL # # # # #
##################################

# Hybrid Recommender System

# 5 movie recommendations

import pandas as pd
pd.set_option('display.max_columns', 500)
movie = pd.read_csv(r'C:\Users\Batuhan\Desktop\recommender_systems\datasets\movie_lens_dataset/movie.csv')
rating = pd.read_csv(r'C:\Users\Batuhan\Desktop\recommender_systems\datasets\movie_lens_dataset/rating.csv')
df = movie.merge(rating, how="left", on="movieId")
df.head()
df["userId"].nunique()


movie = pd.read_csv(r'C:\Users\Batuhan\Desktop\recommender_systems\datasets\movie_lens_dataset/movie.csv')
rating = pd.read_csv(r'C:\Users\Batuhan\Desktop\recommender_systems\datasets\movie_lens_dataset/rating.csv')


df = movie.merge(rating, how="left", on="movieId")
df.head()

comment_counts = pd.DataFrame(df["title"].value_counts())

rare_movies = comment_counts[comment_counts["title"] <= 1000].index

common_movies = df[~df["title"].isin(rare_movies)]

common_movies.shape
common_movies.head()
common_movies["title"].nunique()

user_movie_df = common_movies.pivot_table(index="userId", columns="title", values="rating")

common_movies.info()

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv(r'C:\Users\Batuhan\Desktop\recommender_systems\datasets\movie_lens_dataset/movie.csv', low_memory=False)
    rating = pd.read_csv(r'C:\Users\Batuhan\Desktop\recommender_systems\datasets\movie_lens_dataset/rating.csv', low_memory=False)
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating").astype("float32")

    return user_movie_df

user_movie_df = create_user_movie_df()

user_movie_df.shape
user_movie_df.columns
user_movie_df.iloc[0:10, 0:10]
user_movie_df.index



# random_user = int(pd.Series(user_movie_df.index).sample(1).values)
random_user = 108170

random_user_df = user_movie_df[user_movie_df.index == random_user]

movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.shape

user_movie_count = movies_watched_df.T.notnull().sum()

perc = len(movies_watched) * 0.6
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                      random_user_df[movies_watched]])

corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)

rating = pd.read_csv(r'C:\Users\Batuhan\Desktop\recommender_systems\datasets\movie_lens_dataset/rating.csv', low_memory=False)
top_users.columns = ["userId", "corr"]
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']


recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)

recommendation = movies_to_be_recommend.merge(movie[["movieId", "title"]])

movie = pd.read_csv(r'C:\Users\Batuhan\Desktop\recommender_systems\datasets\movie_lens_dataset/movie.csv', low_memory=False)
rating = pd.read_csv(r'C:\Users\Batuhan\Desktop\recommender_systems\datasets\movie_lens_dataset/rating.csv', low_memory=False)
df = movie.merge(rating, how="left", on="movieId")

movie_id = df[(df["userId"] == random_user) & (df["rating"] == 5.0)]. \
    sort_values("timestamp", ascending=False)["movieId"].values[0]

movie_name = user_movie_df[movie_name]
movie[movie["movieId"] == movie_id]["title"].values[0]
movie_df = user_movie_df[movie[movie["movieId"] == movie_id]["title"].values[0]]
movie_df

corr_movies = user_movie_df.corrwith(movie_df).sort_values(ascending=False)

# 5 movie recommendations

top_5_rec_movies = corr_movies[1:6]
