import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.width", 1000)
pd.set_option('display.max_rows', None)
pd.set_option("display.max_columns", None)

# Soru 1: persona.csv dosyasýný okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.
df = pd.read_csv("persona.csv")
df.head()
df.info()
df.ndim
df.shape
df.dtypes
df.describe().T

# Soru 2: Kaç unique SOURCE vardýr? Frekanslarý nedir?
df["SOURCE"].nunique()
df["SOURCE"].value_counts()
# Soru 3: Kaç unique PRICE vardýr?
df["PRICE"].nunique()
df["PRICE"].unique()
# Soru 4: Hangi PRICE'dan kaçar tane satýþ gerçekleþmiþ?
df["PRICE"].value_counts()
# Soru 5: Hangi ülkeden kaçar tane satýþ olmuþ?
df["COUNTRY"].value_counts()
# Soru 6: Ülkelere göre satýþlardan toplam ne kadar kazanýlmýþ?
df.groupby("COUNTRY")["PRICE"].sum()
# Soru 7: SOURCE türlerine göre satýþ sayýlarý nedir?
df["SOURCE"].value_counts()
# Soru 8: Ülkelere göre PRICE ortalamalarý nedir?
df.groupby("COUNTRY")["PRICE"].mean()
# Soru 9: SOURCE'lara göre PRICE ortalamalarý nedir?
df.groupby("SOURCE")["PRICE"].mean()
# Soru 10: COUNTRY-SOURCE kýrýlýmýnda PRICE ortalamalarý nedir?
df.pivot_table(index="SOURCE", columns="COUNTRY", values="PRICE", aggfunc="mean")

# GÖREV 2: COUNTRY, SOURCE, SEX, AGE kýrýlýmýnda ortalama kazançlar nedir?
df.pivot_table(index=["COUNTRY", "SOURCE", "SEX", "AGE"], aggfunc="mean").head()

# GÖREV 3: Çýktýyý PRICE'a göre sýralayýnýz.
agg_df = df.pivot_table(index=["COUNTRY", "SOURCE", "SEX", "AGE"], aggfunc="mean")
agg_df.sort_values(by="PRICE", ascending=False, inplace=True)
agg_df.head()

# GÖREV 4: Indekste yer alan isimleri deðiþken ismine çeviriniz.
agg_df.reset_index(inplace=True)
agg_df.head()
agg_df.columns

# GÖREV 5: AGE deðiþkenini kategorik deðiþkene çeviriniz ve agg_df'e ekleyiniz.?
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
# GÖREV 6: Yeni level based müþterileri tanýmlayýnýz ve veri setine deðiþken olarak ekleyiniz.
#############################################
# customers_level_based adýnda bir deðiþken tanýmlayýnýz ve veri setine bu deðiþkeni ekleyiniz.
# Dikkat!
# list comp ile customers_level_based deðerleri oluþturulduktan sonra bu deðerlerin tekilleþtirilmesi gerekmektedir.
# Örneðin birden fazla þu ifadeden olabilir: USA_ANDROID_MALE_0_18
# Bunlarý groupby'a alýp price ortalamalarýný almak gerekmektedir.

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
# GÖREV 7: Yeni müþterileri (USA_ANDROID_MALE_0_18) segmentlere ayýrýnýz.
#############################################
# PRICE'a göre segmentlere ayýrýnýz,
# segmentleri "SEGMENT" isimlendirmesi ile agg_df'e ekleyiniz,
# segmentleri betimleyiniz,


new_df2["SEGMENT"] = pd.cut(new_df2["PRICE"], 4, labels=["D", "C", "B", "A"])
new_df2.head()

new_df2.groupby("SEGMENT")["PRICE"].agg(["mean", "max", "sum"])

#############################################
# GÖREV 8: Yeni gelen müþterileri sýnýflandýrýnýz ne kadar gelir getirebileceðini tahmin ediniz.
#############################################
# 33 yaþýnda ANDROID kullanan bir Türk kadýný hangi segmente aittir ve ortalama ne kadar gelir kazandýrmasý beklenir?
# 35 yaþýnda IOS kullanan bir Fransýz kadýný hangi segmente ve ortalama ne kadar gelir kazandýrmasý beklenir?

new_user = "TUR_ANDROID_FEMALE_26_35"

new_df2[new_df2["customers_level_based"] == new_user] # ???

new_user_2 = "FRA_IOS_FEMALE_26_35"

new_df2[new_df2["customers_level_based"] == new_user_2] # ???


