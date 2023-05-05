##################################
# # # # # BATUHAN Y�KSEL # # # # #
##################################

# Hybrid Recommender System

# �� Problemi
# ID'si verilen kullan�c� i�in item-based ve user-based recommender y�ntemlerini kullanarak 10 film �nerisi yap�n�z.

# Veri Seti Hikayesi
# Veri seti, bir film tavsiye hizmeti olan MovieLens taraf�ndan sa�lanm��t�r. ��erisinde filmler ile birlikte bu filmlere yap�lan
# derecelendirme puanlar�n� bar�nd�rmaktad�r. 27.278 filmde 2.000.0263 derecelendirme i�ermektedir. Bu veri seti ise 17 Ekim 2016 tarihinde
# olu�turulmu�tur. 138.493 kullan�c� ve 9 Ocak 1995 ile 31 Mart 2015 tarihleri aras�nda verileri i�ermektedir. Kullan�c�lar rastgele
# se�ilmi�tir. Se�ilen t�m kullan�c�lar�n en az 20 filme oy verdi�i bilgisi mevcuttur.

import pandas as pd
pd.set_option('display.max_columns', 500)
movie = pd.read_csv(r'C:\Users\Batuhan\Desktop\recommender_systems\datasets\movie_lens_dataset/movie.csv')
rating = pd.read_csv(r'C:\Users\Batuhan\Desktop\recommender_systems\datasets\movie_lens_dataset/rating.csv')
df = movie.merge(rating, how="left", on="movieId")
df.head()
df["userId"].nunique()
# Proje G�revleri

# User Based Recommendation

# G�REV 1: Veri Haz�rlama

# Ad�m 1: movie, rating veri setlerini okutunuz

movie = pd.read_csv(r'C:\Users\Batuhan\Desktop\recommender_systems\datasets\movie_lens_dataset/movie.csv')
rating = pd.read_csv(r'C:\Users\Batuhan\Desktop\recommender_systems\datasets\movie_lens_dataset/rating.csv')

# Ad�m 2: rating veri setine Id�lere ait film isimlerini ve t�r�n� movie veri setinden ekleyiniz.

df = movie.merge(rating, how="left", on="movieId")
df.head()

# Ad�m 3: Toplam oy kullan�lma say�s� 1000'in alt�nda olan filmlerin isimlerini listede tutunuz ve veri setinden ��kart�n�z.


comment_counts = pd.DataFrame(df["title"].value_counts())

rare_movies = comment_counts[comment_counts["title"] <= 1000].index

common_movies = df[~df["title"].isin(rare_movies)]

common_movies.shape
common_movies.head()
common_movies["title"].nunique()

# Ad�m 4: index'te userID'lerin sutunlarda film isimlerinin ve de�er olarak ratinglerin bulundu�u dataframe i�in pivot table olu�turunuz.

user_movie_df = common_movies.pivot_table(index="userId", columns="title", values="rating")


# Ad�m 5: Yap�lan t�m i�lemleri fonksiyonla�t�r�n�z.
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

# G�REV 2:  �neri Yap�lacak Kullan�c�n�n �zledi�i Filmlerin Belirlenmesi

# Ad�m 1:Rastgele bir kullan�c� id�si se�iniz.

# random_user = int(pd.Series(user_movie_df.index).sample(1).values)
random_user = 108170

# Ad�m 2: Se�ilen kullan�c�ya ait g�zlem birimlerinde nolu�an random_user_df ad�nda yeni bir dataframe olu�turunuz.

random_user_df = user_movie_df[user_movie_df.index == random_user]


# Ad�m 3: Se�ilen kullan�c�lar�n oy kulland��� filmleri movies_watched ad�nda bir listeye atay�n�z.

movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

# G�rev 3:  Ayn� Filmleri �zleyen Di�er Kullan�c�lar�n Verisine ve Id'lerine Eri�ilmesi

# Ad�m 1: Se�ilen kullan�c�n�n izledi�i fimlere ait sutunlar� user_movie_df'ten se�iniz ve movies_watched_df ad�nda yeni bir
# dataframe olu�turunuz.

movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.shape

# Ad�m 2: Her bir kullanc�n�n se�ili user'in izledi�i filmlerin ka��n� izledi�ini bilgisini ta��yan user_movie_count ad�nda yeni bir
# dataframe olu�turunuz.

user_movie_count = movies_watched_df.T.notnull().sum()

# Ad�m3: Se�ilen kullan�c�n�n oy verdi�i filmlerin y�zde 60 ve �st�n� izleyenlerin kullan�c� id�lerinden users_same_movies ad�nda bir
# liste olu�turunuz.

perc = len(movies_watched) * 0.6
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]


# G�REV 4:  �neri Yap�lacak Kullan�c� ile En Benzer Kullan�c�lar�n Belirlenmesi

# Ad�m 1: user_same_movies listesi i�erisindeki se�ili user ile benzerlik g�steren kullan�c�lar�n id�lerinin bulunaca�� �ekilde
# movies_watched_df dataframe�ini filtreleyiniz.

final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                      random_user_df[movies_watched]])

# Ad�m 2: Kullan�c�lar�n birbirleri ile olan korelasyonlar�n�n bulunaca�� yeni bir corr_df dataframe�i olu�turunuz.

corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()

# Ad�m 3: Se�ili kullan�c� ile y�ksek korelasyona sahip(0.65�in �zerinde olan) kullan�c�lar� filtreleyerek top_users ad�nda yeni bir
# dataframe olu�turunuz.

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)

# Ad�m 4:  top_users dataframe�ine rating veri seti ile merge ediniz.

rating = pd.read_csv(r'C:\Users\Batuhan\Desktop\recommender_systems\datasets\movie_lens_dataset/rating.csv', low_memory=False)
top_users.columns = ["userId", "corr"]
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]

# G�REV 5:  Weighted Average Recommendation Score'un Hesaplanmas� ve �lk 5 Filmin Tutulmas�

# Ad�m 1: Her bir kullan�c�n�n corr ve rating de�erlerinin �arp�m�ndan olu�an weighted_rating ad�nda yeni bir de�i�ken olu�turunuz.

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

# Ad�m 2: Film id�si ve her bir filme ait t�m kullan�c�lar�n weighted rating�lerinin ortalama de�erini i�eren recommendation_df ad�nda
# yeni bir dataframe olu�turunuz.

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()

# Ad�m 3: recommendation_df i�erisinde weighted rating'i 3.5'ten b�y�k olan filmleri se�iniz ve weighted rating�e g�re s�ralay�n�z.

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)

# Ad�m 4: movie veri setinden film isimlerini getiriniz ve tavsiye edilecek ilk 5 filmi se�iniz.

recommendation = movies_to_be_recommend.merge(movie[["movieId", "title"]])

# Item Based Recommendation

# G�REV 1:  Kullan�c�n�n izledi�i en son ve en y�ksek puan verdi�i filme g�re item-based �neri yap�n�z.

# Ad�m 1: movie, rating veri setlerini okutunuz.

movie = pd.read_csv(r'C:\Users\Batuhan\Desktop\recommender_systems\datasets\movie_lens_dataset/movie.csv', low_memory=False)
rating = pd.read_csv(r'C:\Users\Batuhan\Desktop\recommender_systems\datasets\movie_lens_dataset/rating.csv', low_memory=False)
df = movie.merge(rating, how="left", on="movieId")

# Ad�m 2: Se�ili kullan�c�n�n 5 puan verdi�i filmlerden puan� en g�ncel olan filmin id'sini al�n�z.

movie_id = df[(df["userId"] == random_user) & (df["rating"] == 5.0)]. \
    sort_values("timestamp", ascending=False)["movieId"].values[0]


# Ad�m 3: User based recommendation b�l�m�nde olu�turulan user_movie_df dataframe�ini se�ilen film id�sine g�re filtreleyiniz.

movie_name = user_movie_df[movie_name]
movie[movie["movieId"] == movie_id]["title"].values[0]
movie_df = user_movie_df[movie[movie["movieId"] == movie_id]["title"].values[0]]
movie_df

# Ad�m 4: Filtrelenen dataframe�i kullanarak se�ili filmle di�er filmlerin korelasyonunu bulunuz ve s�ralay�n�z.

corr_movies = user_movie_df.corrwith(movie_df).sort_values(ascending=False)

# Ad�m 5: Se�ili film�in kendisi haricinde ilk 5 filmi �neri olarak veriniz.


top_5_rec_movies = corr_movies[1:6]