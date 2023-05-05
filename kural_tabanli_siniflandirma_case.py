import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.width", 1000)
pd.set_option('display.max_rows', None)
pd.set_option("display.max_columns", None)

# Soru 1: persona.csv dosyas�n� okutunuz ve veri seti ile ilgili genel bilgileri g�steriniz.
df = pd.read_csv("persona.csv")
df.head()
df.info()
df.ndim
df.shape
df.dtypes
df.describe().T

# Soru 2: Ka� unique SOURCE vard�r? Frekanslar� nedir?
df["SOURCE"].nunique()
df["SOURCE"].value_counts()
# Soru 3: Ka� unique PRICE vard�r?
df["PRICE"].nunique()
df["PRICE"].unique()
# Soru 4: Hangi PRICE'dan ka�ar tane sat�� ger�ekle�mi�?
df["PRICE"].value_counts()
# Soru 5: Hangi �lkeden ka�ar tane sat�� olmu�?
df["COUNTRY"].value_counts()
# Soru 6: �lkelere g�re sat��lardan toplam ne kadar kazan�lm��?
df.groupby("COUNTRY")["PRICE"].sum()
# Soru 7: SOURCE t�rlerine g�re sat�� say�lar� nedir?
df["SOURCE"].value_counts()
# Soru 8: �lkelere g�re PRICE ortalamalar� nedir?
df.groupby("COUNTRY")["PRICE"].mean()
# Soru 9: SOURCE'lara g�re PRICE ortalamalar� nedir?
df.groupby("SOURCE")["PRICE"].mean()
# Soru 10: COUNTRY-SOURCE k�r�l�m�nda PRICE ortalamalar� nedir?
df.pivot_table(index="SOURCE", columns="COUNTRY", values="PRICE", aggfunc="mean")

# G�REV 2: COUNTRY, SOURCE, SEX, AGE k�r�l�m�nda ortalama kazan�lar nedir?
df.pivot_table(index=["COUNTRY", "SOURCE", "SEX", "AGE"], aggfunc="mean").head()

# G�REV 3: ��kt�y� PRICE'a g�re s�ralay�n�z.
agg_df = df.pivot_table(index=["COUNTRY", "SOURCE", "SEX", "AGE"], aggfunc="mean")
agg_df.sort_values(by="PRICE", ascending=False, inplace=True)
agg_df.head()

# G�REV 4: Indekste yer alan isimleri de�i�ken ismine �eviriniz.
agg_df.reset_index(inplace=True)
agg_df.head()
agg_df.columns

# G�REV 5: AGE de�i�kenini kategorik de�i�kene �eviriniz ve agg_df'e ekleyiniz.?
agg_df["AGE_CAT"] = pd.cut(x=agg_df["AGE"], bins=[0,19,25,35,45,70])
agg_df.head()



liste = []
for i in agg_df["AGE"]:
    if i < 20:
        liste.append("0_19")
    elif 20 <= i < 26:
        liste.append("20_25")
    elif 26 <= i < 36:
        liste.append("26_35")
    elif 36 <= i < 46:
        liste.append("36_45")
    else:
        liste.append("46_70")
liste[0:20]
agg_df["AGE"].head(10)
agg_df.head(20)

agg_df["AGE_CAT_2"] = liste




#############################################
# G�REV 6: Yeni level based m��terileri tan�mlay�n�z ve veri setine de�i�ken olarak ekleyiniz.
#############################################
# customers_level_based ad�nda bir de�i�ken tan�mlay�n�z ve veri setine bu de�i�keni ekleyiniz.
# Dikkat!
# list comp ile customers_level_based de�erleri olu�turulduktan sonra bu de�erlerin tekille�tirilmesi gerekmektedir.
# �rne�in birden fazla �u ifadeden olabilir: USA_ANDROID_MALE_0_18
# Bunlar� groupby'a al�p price ortalamalar�n� almak gerekmektedir.

liste = list(zip(agg_df["COUNTRY"],agg_df["SOURCE"], agg_df["SEX"], agg_df["AGE_CAT_2"]))
agg_df["customers_level_based"] = [i[0].upper() + "_" + i[1].upper() + "_" + i[2].upper() + "_" + i[3] for i in liste]
agg_df.head(20)

new_df = agg_df[["customers_level_based", "PRICE"]]
new_df.head()

new_df2 = new_df.groupby("customers_level_based")["PRICE"].mean()
new_df2 = pd.DataFrame(new_df2)
new_df2.head()
new_df2.reset_index(inplace=True)
new_df2.columns



#############################################
# G�REV 7: Yeni m��terileri (USA_ANDROID_MALE_0_18) segmentlere ay�r�n�z.
#############################################
# PRICE'a g�re segmentlere ay�r�n�z,
# segmentleri "SEGMENT" isimlendirmesi ile agg_df'e ekleyiniz,
# segmentleri betimleyiniz,


new_df2["SEGMENT"] = pd.cut(new_df2["PRICE"], 4, labels=["D", "C", "B", "A"])
new_df2.head()

new_df2.groupby("SEGMENT")["PRICE"].agg(["mean", "max", "sum"])

#############################################
# G�REV 8: Yeni gelen m��terileri s�n�fland�r�n�z ne kadar gelir getirebilece�ini tahmin ediniz.
#############################################
# 33 ya��nda ANDROID kullanan bir T�rk kad�n� hangi segmente aittir ve ortalama ne kadar gelir kazand�rmas� beklenir?
# 35 ya��nda IOS kullanan bir Frans�z kad�n� hangi segmente ve ortalama ne kadar gelir kazand�rmas� beklenir?

new_user = "TUR_ANDROID_FEMALE_26_35"

new_df2[new_df2["customers_level_based"] == new_user] # ???

new_user_2 = "FRA_IOS_FEMALE_26_35"

new_df2[new_df2["customers_level_based"] == new_user_2] # ???


