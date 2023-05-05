# # # RATING PRODUCTS & SORTING REVIEWS IN AMAZON

# �� PROBLEM�

# E-ticaretteki en �nemli problemlerden bir tanesi �r�nlere sat�� sonras� verilen puanlar�n do�ru �ekilde hesaplanmas�d�r. Bu problemin
# ��z�m� e-ticaret sitesi i�in daha fazla m��teri memnuniyeti sa�lamak, sat�c�lar i�in �r�n�n �ne ��kmas� ve sat�n alanlar i�in sorunsuz
# bir al��veri� deneyimi demektir. Bir di�er problem ise �r�nlere verilen yorumlar�n do�ru bir �ekilde s�ralanmas� olarak kar��m�za ��kmaktad�r.
# Yan�lt�c� yorumlar�n �ne ��kmas� �r�n�n sat���n� do�rudan etkileyece�inden dolay� hem maddi kay�p hem de m��teri kayb�na neden olacakt�r.
# Bu 2 temel problemin ��z�m�nde e-ticaret sitesi ve sat�c�lar sat��lar�n� artt�r�rken m��teriler ise sat�n alma yolculu�unu sorunsuz olarak
# tamamlayacakt�r.

# Amazon �r�n verilerini i�eren bu veri seti �r�n kategorileri ile �e�itli metadatalar� i�ermektedir. Elektronik kategorisindeki
# en fazla yorum alan �r�n�n kullan�c� puanlar� ve yorumlar� vard�r.

# reviewerID: Kullan�c� ID�si
# asin: �r�n ID�si
# reviewerName: Kullan�c� Ad�
# helpful: Faydal� de�erlendirme derecesi
# reviewText: De�erlendirme
# overall: �r�n rating�i
# summary: De�erlendirme �zeti
# unixReviewTime: De�erlendirme zaman�
# reviewTime: De�erlendirme zaman� Raw
# day_diff: De�erlendirmeden itibaren ge�en g�n say�s�
# helpful_yes: De�erlendirmenin faydal� bulunma say�s�
# total_vote: De�erlendirmeye verilen oy say�s�

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv(r"C:\Users\Batuhan\Desktop\amazon_review.csv")
df.shape
df.info()
df.head()
# df.isnull().values.any()
# df.isnull().sum()
# df["asin"].unique()
# df.dropna(inplace=True)
# # # Proje G�revleri # # #

# G�REV 1: Average Rating�i g�ncel yorumlara g�re hesaplay�n�z ve var olan average rating ile k�yaslay�n�z.
# Payla��lan veri setinde kullan�c�lar bir �r�ne puanlar vermi� ve yorumlar yapm��t�r. Bu g�revde amac�m�z verilen puanlar� tarihe g�re
# a��rl�kland�rarak de�erlendirmek. �lk ortalama puan ile elde edilecek tarihe g�re a��rl�kl� puan�n kar��la�t�r�lmas� gerekmektedir.

# Ad�m 1: �r�n�n ortalama puan�n� hesaplay�n�z.

df["overall"].mean()

# Ad�m 2: Tarihe g�re a��rl�kl� puan ortalamas�n� hesaplay�n�z.

df["reviewTime"].max()
df["day_diff"].max()
df["day_diff"].describe().T


df.loc[df["day_diff"] <= 30, "overall"].mean() * 0.22 + \
    df.loc[(df["day_diff"] > 30) & (df["day_diff"] <= 90), "overall"].mean() * 0.20 + \
    df.loc[(df["day_diff"] > 90) & (df["day_diff"] <= 180), "overall"].mean() * 0.18 + \
    df.loc[(df["day_diff"] > 180) & (df["day_diff"] <= 360), "overall"].mean() * 0.16 + \
    df.loc[(df["day_diff"] > 360) & (df["day_diff"] <= 720), "overall"].mean() * 0.14 + \
    df.loc[(df["day_diff"] > 720) & (df["day_diff"] <= np.inf), "overall"].mean() * 0.10


# Ad�m 3: A��rl�kland�r�lm�� puanlamada her bir zaman diliminin ortalamas�n� kar��la�t�r�p yorumlay�n�z. ?
"""
df.loc[df["day_diff"] <= 30, "overall"].mean() * 0.22
df.loc[(df["day_diff"] > 30) & (df["day_diff"] <= 90), "overall"].mean() * 0.20
df.loc[(df["day_diff"] > 90) & (df["day_diff"] <= 180), "overall"].mean() * 0.18
df.loc[(df["day_diff"] > 180) & (df["day_diff"] <= 360), "overall"].mean() * 0.16
df.loc[(df["day_diff"] > 360) & (df["day_diff"] <= 720), "overall"].mean() * 0.14
df.loc[(df["day_diff"] > 720) & (df["day_diff"] <= np.inf), "overall"].mean() * 0.10
"""


x = df["overall"].mean() # 4.587589013224822

a = df.loc[df["day_diff"] <= 30, "overall"].mean() # 4.742424242424242 * 0.22
b = df.loc[(df["day_diff"] > 30) & (df["day_diff"] <= 90), "overall"].mean() # 4.803149606299213 * 0.20
c = df.loc[(df["day_diff"] > 90) & (df["day_diff"] <= 180), "overall"].mean() # 4.649484536082475 * 0.18
d = df.loc[(df["day_diff"] > 180) & (df["day_diff"] <= 360), "overall"].mean() # 4.681021897810219 * 0.16
e = df.loc[(df["day_diff"] > 360) & (df["day_diff"] <= 720), "overall"].mean() # 4.54839953721558 * 0.14
f = df.loc[(df["day_diff"] > 720) & (df["day_diff"] <= np.inf), "overall"].mean() # 4.350404312668464 * 0.10


df["cat"] = pd.cut(df["day_diff"], bins=[0, 30, 90, 180, 360, 720, np.inf], labels=["A", "B", "C", "D", "E", "F"])

df2 = df.groupby("cat")["overall"].mean()
x = [x] * 6
x = pd.DataFrame(x, index=["A", "B", "C", "D", "E", "F"], columns=[""])
df3 = pd.concat([df2, x], axis=1)

plt.plot(df3.iloc[:,0], "r-")
plt.plot(df3.iloc[:,1], "b--")
plt.xlabel("D�nemler")
plt.ylabel("Ortalama")
plt.title("Genel Ortalama vs D�nemsel Ortalama")

# Yorum:

# G�REV 2: �r�n i�in �r�n detay sayfas�nda g�r�nt�lenecek 20 review'i belirleyiniz.

# Ad�m 1: helpful_no de�i�kenini �retiniz.

# � total_vote bir yoruma verilen toplam up-down say�s�d�r.
# � up, helpful demektir.
# � Veri setinde helpful_no de�i�keni yoktur, var olan de�i�kenler �zerinden �retilmesi gerekmektedir.
# � Toplam oy say�s�ndan (total_vote) yararl� oy say�s� (helpful_yes) ��kar�larak yararl� bulunmayan oy say�lar�n� (helpful_no) bulunuz.

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]
df.helpful_no.min()
df.head()

# Ad�m 2: score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlar�n� hesaplay�p veriye ekleyiniz.
# � score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlar�n� hesaplayabilmek i�in score_pos_neg_diff,
# score_average_rating ve wilson_lower_bound fonksiyonlar�n� tan�mlay�n�z.
# � score_pos_neg_diff'a g�re skorlar olu�turunuz. Ard�ndan; df i�erisinde score_pos_neg_diff ismiyle kaydediniz.
# � score_average_rating'a g�re skorlar olu�turunuz. Ard�ndan; df i�erisinde score_average_rating ismiyle kaydediniz.
# � wilson_lower_bound'a g�re skorlar olu�turunuz. Ard�ndan; df i�erisinde wilson_lower_bound ismiyle kaydediniz.

df["score_pos_neg_diff"] = df["helpful_yes"] - df["helpful_no"]
df["score_pos_neg_diff"].min()
df["helpful_no"].max()
df["helpful_yes"].min()


def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"],x["helpful_no"]), axis=1)
df.head()
df["score_average_rating"].max()
df["score_average_rating"].describe()



def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)
df.head()
df["wilson_lower_bound"].describe()


# Ad�m 3: 20 Yorumu belirleyiniz ve sonu�lar� Yorumlay�n�z.

# � wilson_lower_bound'a g�re ilk 20 yorumu belirleyip s�ralayan�z.

df.sort_values("wilson_lower_bound", ascending=False).head(20)

# � Sonu�lar� yorumlay�n�z.
df[["total_vote","helpful_yes","helpful_no"]].describe([0.01,0.25,0.5,0.75,0.9,0.95,0.99,0.995,0.999]).T
# Yorumlar�n %99'u 4 oydan daha az oy alm��t�r. Bir yoruma verilen oy say�s� pozitif ya da negatif yorumun �ne ��kmas�na sebep olmu�tur.
# sadece pozitif oyu olup negatifi olmayan yorumlar s�ralamada negatifi olanlar�n da gerisine d��ebilmi�tir.
# Bir yorumun �ne ��kmas� i�in oy say�s� �nemli bir etkendir
