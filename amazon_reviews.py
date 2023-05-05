# # # RATING PRODUCTS & SORTING REVIEWS IN AMAZON

# ÝÞ PROBLEMÝ

# E-ticaretteki en önemli problemlerden bir tanesi ürünlere satýþ sonrasý verilen puanlarýn doðru þekilde hesaplanmasýdýr. Bu problemin
# çözümü e-ticaret sitesi için daha fazla müþteri memnuniyeti saðlamak, satýcýlar için ürünün öne çýkmasý ve satýn alanlar için sorunsuz
# bir alýþveriþ deneyimi demektir. Bir diðer problem ise ürünlere verilen yorumlarýn doðru bir þekilde sýralanmasý olarak karþýmýza çýkmaktadýr.
# Yanýltýcý yorumlarýn öne çýkmasý ürünün satýþýný doðrudan etkileyeceðinden dolayý hem maddi kayýp hem de müþteri kaybýna neden olacaktýr.
# Bu 2 temel problemin çözümünde e-ticaret sitesi ve satýcýlar satýþlarýný arttýrýrken müþteriler ise satýn alma yolculuðunu sorunsuz olarak
# tamamlayacaktýr.

# Amazon ürün verilerini içeren bu veri seti ürün kategorileri ile çeþitli metadatalarý içermektedir. Elektronik kategorisindeki
# en fazla yorum alan ürünün kullanýcý puanlarý ve yorumlarý vardýr.

# reviewerID: Kullanýcý ID’si
# asin: Ürün ID’si
# reviewerName: Kullanýcý Adý
# helpful: Faydalý deðerlendirme derecesi
# reviewText: Deðerlendirme
# overall: Ürün rating’i
# summary: Deðerlendirme özeti
# unixReviewTime: Deðerlendirme zamaný
# reviewTime: Deðerlendirme zamaný Raw
# day_diff: Deðerlendirmeden itibaren geçen gün sayýsý
# helpful_yes: Deðerlendirmenin faydalý bulunma sayýsý
# total_vote: Deðerlendirmeye verilen oy sayýsý

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
# # # Proje Görevleri # # #

# GÖREV 1: Average Rating’i güncel yorumlara göre hesaplayýnýz ve var olan average rating ile kýyaslayýnýz.
# Paylaþýlan veri setinde kullanýcýlar bir ürüne puanlar vermiþ ve yorumlar yapmýþtýr. Bu görevde amacýmýz verilen puanlarý tarihe göre
# aðýrlýklandýrarak deðerlendirmek. Ýlk ortalama puan ile elde edilecek tarihe göre aðýrlýklý puanýn karþýlaþtýrýlmasý gerekmektedir.

# Adým 1: Ürünün ortalama puanýný hesaplayýnýz.

df["overall"].mean()

# Adým 2: Tarihe göre aðýrlýklý puan ortalamasýný hesaplayýnýz.

df["reviewTime"].max()
df["day_diff"].max()
df["day_diff"].describe().T


df.loc[df["day_diff"] <= 30, "overall"].mean() * 0.22 + \
    df.loc[(df["day_diff"] > 30) & (df["day_diff"] <= 90), "overall"].mean() * 0.20 + \
    df.loc[(df["day_diff"] > 90) & (df["day_diff"] <= 180), "overall"].mean() * 0.18 + \
    df.loc[(df["day_diff"] > 180) & (df["day_diff"] <= 360), "overall"].mean() * 0.16 + \
    df.loc[(df["day_diff"] > 360) & (df["day_diff"] <= 720), "overall"].mean() * 0.14 + \
    df.loc[(df["day_diff"] > 720) & (df["day_diff"] <= np.inf), "overall"].mean() * 0.10


# Adým 3: Aðýrlýklandýrýlmýþ puanlamada her bir zaman diliminin ortalamasýný karþýlaþtýrýp yorumlayýnýz. ?
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
plt.xlabel("Dönemler")
plt.ylabel("Ortalama")
plt.title("Genel Ortalama vs Dönemsel Ortalama")

# Yorum:

# GÖREV 2: Ürün için ürün detay sayfasýnda görüntülenecek 20 review'i belirleyiniz.

# Adým 1: helpful_no deðiþkenini üretiniz.

# • total_vote bir yoruma verilen toplam up-down sayýsýdýr.
# • up, helpful demektir.
# • Veri setinde helpful_no deðiþkeni yoktur, var olan deðiþkenler üzerinden üretilmesi gerekmektedir.
# • Toplam oy sayýsýndan (total_vote) yararlý oy sayýsý (helpful_yes) çýkarýlarak yararlý bulunmayan oy sayýlarýný (helpful_no) bulunuz.

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]
df.helpful_no.min()
df.head()

# Adým 2: score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarýný hesaplayýp veriye ekleyiniz.
# • score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarýný hesaplayabilmek için score_pos_neg_diff,
# score_average_rating ve wilson_lower_bound fonksiyonlarýný tanýmlayýnýz.
# • score_pos_neg_diff'a göre skorlar oluþturunuz. Ardýndan; df içerisinde score_pos_neg_diff ismiyle kaydediniz.
# • score_average_rating'a göre skorlar oluþturunuz. Ardýndan; df içerisinde score_average_rating ismiyle kaydediniz.
# • wilson_lower_bound'a göre skorlar oluþturunuz. Ardýndan; df içerisinde wilson_lower_bound ismiyle kaydediniz.

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


# Adým 3: 20 Yorumu belirleyiniz ve sonuçlarý Yorumlayýnýz.

# • wilson_lower_bound'a göre ilk 20 yorumu belirleyip sýralayanýz.

df.sort_values("wilson_lower_bound", ascending=False).head(20)

# • Sonuçlarý yorumlayýnýz.
df[["total_vote","helpful_yes","helpful_no"]].describe([0.01,0.25,0.5,0.75,0.9,0.95,0.99,0.995,0.999]).T
# Yorumlarýn %99'u 4 oydan daha az oy almýþtýr. Bir yoruma verilen oy sayýsý pozitif ya da negatif yorumun öne çýkmasýna sebep olmuþtur.
# sadece pozitif oyu olup negatifi olmayan yorumlar sýralamada negatifi olanlarýn da gerisine düþebilmiþtir.
# Bir yorumun öne çýkmasý için oy sayýsý önemli bir etkendir
