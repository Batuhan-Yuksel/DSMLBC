##################################
# # # # # BATUHAN YÜKSEL # # # # #
##################################

# Hybrid Recommender System

# Ýþ Problemi
# ID'si verilen kullanýcý için item-based ve user-based recommender yöntemlerini kullanarak 10 film önerisi yapýnýz.

# Veri Seti Hikayesi
# Veri seti, bir film tavsiye hizmeti olan MovieLens tarafýndan saðlanmýþtýr. Ýçerisinde filmler ile birlikte bu filmlere yapýlan
# derecelendirme puanlarýný barýndýrmaktadýr. 27.278 filmde 2.000.0263 derecelendirme içermektedir. Bu veri seti ise 17 Ekim 2016 tarihinde
# oluþturulmuþtur. 138.493 kullanýcý ve 9 Ocak 1995 ile 31 Mart 2015 tarihleri arasýnda verileri içermektedir. Kullanýcýlar rastgele
# seçilmiþtir. Seçilen tüm kullanýcýlarýn en az 20 filme oy verdiði bilgisi mevcuttur.

import pandas as pd
pd.set_option('display.max_columns', 500)
movie = pd.read_csv(r'C:\Users\Batuhan\Desktop\recommender_systems\datasets\movie_lens_dataset/movie.csv')
rating = pd.read_csv(r'C:\Users\Batuhan\Desktop\recommender_systems\datasets\movie_lens_dataset/rating.csv')
df = movie.merge(rating, how="left", on="movieId")
df.head()
df["userId"].nunique()
# Proje Görevleri

# User Based Recommendation

# GÖREV 1: Veri Hazýrlama

# Adým 1: movie, rating veri setlerini okutunuz

movie = pd.read_csv(r'C:\Users\Batuhan\Desktop\recommender_systems\datasets\movie_lens_dataset/movie.csv')
rating = pd.read_csv(r'C:\Users\Batuhan\Desktop\recommender_systems\datasets\movie_lens_dataset/rating.csv')

# Adým 2: rating veri setine Id’lere ait film isimlerini ve türünü movie veri setinden ekleyiniz.

df = movie.merge(rating, how="left", on="movieId")
df.head()

# Adým 3: Toplam oy kullanýlma sayýsý 1000'in altýnda olan filmlerin isimlerini listede tutunuz ve veri setinden çýkartýnýz.


comment_counts = pd.DataFrame(df["title"].value_counts())

rare_movies = comment_counts[comment_counts["title"] <= 1000].index

common_movies = df[~df["title"].isin(rare_movies)]

common_movies.shape
common_movies.head()
common_movies["title"].nunique()

# Adým 4: index'te userID'lerin sutunlarda film isimlerinin ve deðer olarak ratinglerin bulunduðu dataframe için pivot table oluþturunuz.

user_movie_df = common_movies.pivot_table(index="userId", columns="title", values="rating")


# Adým 5: Yapýlan tüm iþlemleri fonksiyonlaþtýrýnýz.
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

# GÖREV 2:  Öneri Yapýlacak Kullanýcýnýn Ýzlediði Filmlerin Belirlenmesi

# Adým 1:Rastgele bir kullanýcý id’si seçiniz.

# random_user = int(pd.Series(user_movie_df.index).sample(1).values)
random_user = 108170

# Adým 2: Seçilen kullanýcýya ait gözlem birimlerinde noluþan random_user_df adýnda yeni bir dataframe oluþturunuz.

random_user_df = user_movie_df[user_movie_df.index == random_user]


# Adým 3: Seçilen kullanýcýlarýn oy kullandýðý filmleri movies_watched adýnda bir listeye atayýnýz.

movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

# Görev 3:  Ayný Filmleri Ýzleyen Diðer Kullanýcýlarýn Verisine ve Id'lerine Eriþilmesi

# Adým 1: Seçilen kullanýcýnýn izlediði fimlere ait sutunlarý user_movie_df'ten seçiniz ve movies_watched_df adýnda yeni bir
# dataframe oluþturunuz.

movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.shape

# Adým 2: Her bir kullancýnýn seçili user'in izlediði filmlerin kaçýný izlediðini bilgisini taþýyan user_movie_count adýnda yeni bir
# dataframe oluþturunuz.

user_movie_count = movies_watched_df.T.notnull().sum()

# Adým3: Seçilen kullanýcýnýn oy verdiði filmlerin yüzde 60 ve üstünü izleyenlerin kullanýcý id’lerinden users_same_movies adýnda bir
# liste oluþturunuz.

perc = len(movies_watched) * 0.6
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]


# GÖREV 4:  Öneri Yapýlacak Kullanýcý ile En Benzer Kullanýcýlarýn Belirlenmesi

# Adým 1: user_same_movies listesi içerisindeki seçili user ile benzerlik gösteren kullanýcýlarýn id’lerinin bulunacaðý þekilde
# movies_watched_df dataframe’ini filtreleyiniz.

final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                      random_user_df[movies_watched]])

# Adým 2: Kullanýcýlarýn birbirleri ile olan korelasyonlarýnýn bulunacaðý yeni bir corr_df dataframe’i oluþturunuz.

corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()

# Adým 3: Seçili kullanýcý ile yüksek korelasyona sahip(0.65’in üzerinde olan) kullanýcýlarý filtreleyerek top_users adýnda yeni bir
# dataframe oluþturunuz.

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)

# Adým 4:  top_users dataframe’ine rating veri seti ile merge ediniz.

rating = pd.read_csv(r'C:\Users\Batuhan\Desktop\recommender_systems\datasets\movie_lens_dataset/rating.csv', low_memory=False)
top_users.columns = ["userId", "corr"]
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]

# GÖREV 5:  Weighted Average Recommendation Score'un Hesaplanmasý ve Ýlk 5 Filmin Tutulmasý

# Adým 1: Her bir kullanýcýnýn corr ve rating deðerlerinin çarpýmýndan oluþan weighted_rating adýnda yeni bir deðiþken oluþturunuz.

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

# Adým 2: Film id’si ve her bir filme ait tüm kullanýcýlarýn weighted rating’lerinin ortalama deðerini içeren recommendation_df adýnda
# yeni bir dataframe oluþturunuz.

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()

# Adým 3: recommendation_df içerisinde weighted rating'i 3.5'ten büyük olan filmleri seçiniz ve weighted rating’e göre sýralayýnýz.

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)

# Adým 4: movie veri setinden film isimlerini getiriniz ve tavsiye edilecek ilk 5 filmi seçiniz.

recommendation = movies_to_be_recommend.merge(movie[["movieId", "title"]])

# Item Based Recommendation

# GÖREV 1:  Kullanýcýnýn izlediði en son ve en yüksek puan verdiði filme göre item-based öneri yapýnýz.

# Adým 1: movie, rating veri setlerini okutunuz.

movie = pd.read_csv(r'C:\Users\Batuhan\Desktop\recommender_systems\datasets\movie_lens_dataset/movie.csv', low_memory=False)
rating = pd.read_csv(r'C:\Users\Batuhan\Desktop\recommender_systems\datasets\movie_lens_dataset/rating.csv', low_memory=False)
df = movie.merge(rating, how="left", on="movieId")

# Adým 2: Seçili kullanýcýnýn 5 puan verdiði filmlerden puaný en güncel olan filmin id'sini alýnýz.

movie_id = df[(df["userId"] == random_user) & (df["rating"] == 5.0)]. \
    sort_values("timestamp", ascending=False)["movieId"].values[0]


# Adým 3: User based recommendation bölümünde oluþturulan user_movie_df dataframe’ini seçilen film id’sine göre filtreleyiniz.

movie_name = user_movie_df[movie_name]
movie[movie["movieId"] == movie_id]["title"].values[0]
movie_df = user_movie_df[movie[movie["movieId"] == movie_id]["title"].values[0]]
movie_df

# Adým 4: Filtrelenen dataframe’i kullanarak seçili filmle diðer filmlerin korelasyonunu bulunuz ve sýralayýnýz.

corr_movies = user_movie_df.corrwith(movie_df).sort_values(ascending=False)

# Adým 5: Seçili film’in kendisi haricinde ilk 5 filmi öneri olarak veriniz.


top_5_rec_movies = corr_movies[1:6]