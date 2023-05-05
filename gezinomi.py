import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

pd.set_option("display.width", 1000)
pd.set_option('display.max_rows', None)
pd.set_option("display.max_columns", None)

# KURAL TABANLI SINIFLANDIRMA ILE POTANSIYEL M��TERI GETIRISI HESAPLAMA

# �� PROBLEM�

# Gezinomi yapt��� sat��lar�n baz� �zelliklerini kullanarak seviye tabanl� (level based) yeni sat�� tan�mlar� olu�turmak ve bu yeni sat��
# tan�mlar�na g�re segmentler olu�turup bu segmentlere g�re yeni  gelebilecek m��terilerin �irkete ortalama ne kadar kazand�rabilece�ini
# tahmin etmek istemektedir.

# �rne�in:
# Antalya�dan Her�ey Dahil bir otele yo�un bir d�nemde gitmek isteyen bir m��terinin ortalama ne kadar kazand�rabilece�i belirlenmek isteniyor.

# Veri Seti Hikayesi

# gezinomi_miuul.xlsx veri seti Gezinomi �irketinin yapt��� sat��lar�n fiyatlar�n� ve bu sat��lara ait bilgiler i�ermektedir.
# Veri seti her sat�� i�leminde olu�an kay�tlardan meydana gelmektedir. Bunun anlam� tablo tekille�tirilmemi�tir. Di�er bir ifade ile
# m��teri birden fazla al��veri� yapm�� olabilir.

# De�i�kenler

# SaleId: Sat�� id
# SaleDate : Sat�� Tarihi
# Price: Sat�� i�in �denen fiyat
# ConceptName:Otel konsept bilgisi
# SaleCityName: Otelin bulundu�u �ehir bilgisi
# CheckInDate: M��terinin otele giri� tarihi
# CInDay:M��terinin otele giri� g�n�
# SaleCheckInDayDiff: Check in ile giri� tarihi g�n fark�
# Season:Otele giri� tarihindeki sezon bilgisi

# Proje G�revleri

# G�REV 1: A�a��daki Sorular� Yan�tlay�n�z

# Soru 1: miuul_gezinomi.xlsx dosyas�n� okutunuz ve veri seti ile ilgili genel bilgileri g�steriniz.

df = pd.read_excel(r"C:\Users\Batuhan\Desktop\miuul_gezinomi.xlsx")
df.shape
df.info()
df.dtypes
df.isnull().values.any()
df.isnull().sum()
df.head()
df.describe().T

# Soru 2: Ka� unique �ehir vard�r? Frekanslar� nedir?

df["SaleCityName"].unique()
df["SaleCityName"].nunique()
df["SaleCityName"].value_counts()

# Soru 3: Ka� unique Concept vard�r?

df["ConceptName"].unique()
df["ConceptName"].nunique()

# Soru 4: Hangi Concept�den ka�ar tane sat�� ger�ekle�mi�?

df["ConceptName"].value_counts()

# Soru 5: �ehirlere g�re sat��lardan toplam ne kadar kazan�lm��?

df.groupby("SaleCityName")["Price"].sum()

# Soru 6: Concept t�rlerine g�re g�re ne kadar kazan�lm��?

df.groupby("ConceptName")["Price"].sum()

# Soru 7: �ehirlere g�re PRICE ortalamalar� nedir?

df.groupby("SaleCityName")["Price"].mean()

# Soru 8: Conceptlere g�re PRICE ortalamalar� nedir?

df.groupby("ConceptName")["Price"].mean()

# Soru 9: �ehir-Concept k�r�l�m�nda PRICE ortalamalar� nedir?

df.pivot_table(index="ConceptName", columns="SaleCityName", values="Price", aggfunc="mean")

# G�REV 2: SaleCheckInDayDiff de�i�kenini kategorik bir de�i�kene �eviriniz.

# � SaleCheckInDayDiff de�i�keni m��terinin CheckIn tarihinden ne kadar �nce satin al�m�n� tamamlad���n� g�sterir.
# � Aral�klar� ikna edici �ekilde olu�turunuz.
#   �rne�in: �0_7�, �7_30', �30_90', �90_max� aral�klar�n� kullanabilirsiniz.
# � Bu aral�klar i�in "Last Minuters", "Potential Planners", "Planners", "Early Bookers� isimlerini kullanabilirsiniz

df["EB Score"] = pd.cut(df["SaleCheckInDayDiff"], bins=[0, 7, 30, 90, df["SaleCheckInDayDiff"].max() + 1], right=False,
                                  labels=["Last Minuters", "Potential Planners", "Planners", "Early Bookers"])

# G�REV 3: COUNTRY, SOURCE, SEX, AGE k�r�l�m�nda ortalama kazan�lar nedir ??????????????????????????????????????????????

# �ehir-Concept-EB Score, �ehir-Concept- Sezon, �ehir-Concept-CInDay k�r�l�m�nda ortalama �denen �cret ve yap�lan i�lem say�s� cinsinden
# inceleyiniz ? ????????????????????????????????????????????????????????????????????

df.pivot_table(index=["SaleCityName", "ConceptName", "EB Score"], aggfunc=["mean","count"]).head()

df.groupby(["SaleCityName", "ConceptName", "EB Score"])["Price"].agg(["mean","count"]).head()

df.groupby(["SaleCityName", "ConceptName", "Seasons"])["Price"].agg(["mean","count"]).head()

df.groupby(["SaleCityName", "ConceptName", "CInDay"])["Price"].agg(["mean","count"]).head()

# G�REV 4: City-Concept-Season k�r�l�m�n�n ��kt�s�n� PRICE'a g�re s�ralay�n�z. ????????????????????????????????????????????????????????

agg_df = df.pivot_table(index=["SaleCityName", "ConceptName", "Seasons"], values="Price").sort_values(by="Price", ascending=False)



# G�REV 5: Indekste yer alan isimleri de�i�ken ismine �eviriniz. ???????????????????????????????????????????????????????????????????

agg_df.reset_index(inplace=True)

# G�REV 6: Yeni seviye tabanl� m��terileri (persona) tan�mlay�n�z  ????????????????????????????????????????????????????????????

# � Yeni seviye tabanl� sat��lar� tan�mlay�n�z ve veri setine de�i�ken olarak ekleyiniz.
# � Yeni eklenecek de�i�kenin ad�: sales_level_based
# � �nceki soruda elde edece�iniz ��kt�daki g�zlemleri bir araya getirerek sales_level_based de�i�kenini olu�turman�z gerekmektedir.

agg_df["sales_level_based"] = agg_df["SaleCityName"].astype(str) + "_" + agg_df["ConceptName"].astype(str) + "_" + \
                              agg_df["Seasons"].astype(str)

# G�rev 7: Yeni m��terileri (personalar�) segmentlere ay�r�n�z.

agg_df["SEGMENT"] = pd.qcut(agg_df["Price"], 4, labels=["D","C","B","A"])

# G�REV 8: Yeni gelen m��terileri s�n�fland�r�p, ne kadar gelir getirebileceklerini tahmin ediniz.

# � Antalya�da her�ey dahil ve y�ksek sezonda tatil yapmak isteyen bir ki�inin ortalama ne kadar gelir kazand�rmas� beklenir?

# � Girne�de yar�m pansiyon bir otele d���k sezonda giden bir tatilci hangi segmentte yer alacakt�r?

user = "Antalya_Her�ey Dahil_High"

agg_df[agg_df["sales_level_based"] == user]["Price"]

user = "Girne_Yar�m Pansiyon_Low"

agg_df[agg_df["sales_level_based"] == user]["Price"]