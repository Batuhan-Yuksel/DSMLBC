import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

pd.set_option("display.width", 1000)
pd.set_option('display.max_rows', None)
pd.set_option("display.max_columns", None)

# KURAL TABANLI SINIFLANDIRMA ILE POTANSIYEL MÜÞTERI GETIRISI HESAPLAMA

# ÝÞ PROBLEMÝ

# Gezinomi yaptýðý satýþlarýn bazý özelliklerini kullanarak seviye tabanlý (level based) yeni satýþ tanýmlarý oluþturmak ve bu yeni satýþ
# tanýmlarýna göre segmentler oluþturup bu segmentlere göre yeni  gelebilecek müþterilerin þirkete ortalama ne kadar kazandýrabileceðini
# tahmin etmek istemektedir.

# Örneðin:
# Antalya’dan Herþey Dahil bir otele yoðun bir dönemde gitmek isteyen bir müþterinin ortalama ne kadar kazandýrabileceði belirlenmek isteniyor.

# Veri Seti Hikayesi

# gezinomi_miuul.xlsx veri seti Gezinomi þirketinin yaptýðý satýþlarýn fiyatlarýný ve bu satýþlara ait bilgiler içermektedir.
# Veri seti her satýþ iþleminde oluþan kayýtlardan meydana gelmektedir. Bunun anlamý tablo tekilleþtirilmemiþtir. Diðer bir ifade ile
# müþteri birden fazla alýþveriþ yapmýþ olabilir.

# Deðiþkenler

# SaleId: Satýþ id
# SaleDate : Satýþ Tarihi
# Price: Satýþ için ödenen fiyat
# ConceptName:Otel konsept bilgisi
# SaleCityName: Otelin bulunduðu þehir bilgisi
# CheckInDate: Müþterinin otele giriþ tarihi
# CInDay:Müþterinin otele giriþ günü
# SaleCheckInDayDiff: Check in ile giriþ tarihi gün farký
# Season:Otele giriþ tarihindeki sezon bilgisi

# Proje Görevleri

# GÖREV 1: Aþaðýdaki Sorularý Yanýtlayýnýz

# Soru 1: miuul_gezinomi.xlsx dosyasýný okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.

df = pd.read_excel(r"C:\Users\Batuhan\Desktop\miuul_gezinomi.xlsx")
df.shape
df.info()
df.dtypes
df.isnull().values.any()
df.isnull().sum()
df.head()
df.describe().T

# Soru 2: Kaç unique þehir vardýr? Frekanslarý nedir?

df["SaleCityName"].unique()
df["SaleCityName"].nunique()
df["SaleCityName"].value_counts()

# Soru 3: Kaç unique Concept vardýr?

df["ConceptName"].unique()
df["ConceptName"].nunique()

# Soru 4: Hangi Concept’den kaçar tane satýþ gerçekleþmiþ?

df["ConceptName"].value_counts()

# Soru 5: Þehirlere göre satýþlardan toplam ne kadar kazanýlmýþ?

df.groupby("SaleCityName")["Price"].sum()

# Soru 6: Concept türlerine göre göre ne kadar kazanýlmýþ?

df.groupby("ConceptName")["Price"].sum()

# Soru 7: Þehirlere göre PRICE ortalamalarý nedir?

df.groupby("SaleCityName")["Price"].mean()

# Soru 8: Conceptlere göre PRICE ortalamalarý nedir?

df.groupby("ConceptName")["Price"].mean()

# Soru 9: Þehir-Concept kýrýlýmýnda PRICE ortalamalarý nedir?

df.pivot_table(index="ConceptName", columns="SaleCityName", values="Price", aggfunc="mean")

# GÖREV 2: SaleCheckInDayDiff deðiþkenini kategorik bir deðiþkene çeviriniz.

# • SaleCheckInDayDiff deðiþkeni müþterinin CheckIn tarihinden ne kadar önce satin alýmýný tamamladýðýný gösterir.
# • Aralýklarý ikna edici þekilde oluþturunuz.
#   Örneðin: ‘0_7’, ‘7_30', ‘30_90', ‘90_max’ aralýklarýný kullanabilirsiniz.
# • Bu aralýklar için "Last Minuters", "Potential Planners", "Planners", "Early Bookers“ isimlerini kullanabilirsiniz

df["EB Score"] = pd.cut(df["SaleCheckInDayDiff"], bins=[0, 7, 30, 90, df["SaleCheckInDayDiff"].max() + 1], right=False,
                                  labels=["Last Minuters", "Potential Planners", "Planners", "Early Bookers"])

# GÖREV 3: COUNTRY, SOURCE, SEX, AGE kýrýlýmýnda ortalama kazançlar nedir ??????????????????????????????????????????????

# Þehir-Concept-EB Score, Þehir-Concept- Sezon, Þehir-Concept-CInDay kýrýlýmýnda ortalama ödenen ücret ve yapýlan iþlem sayýsý cinsinden
# inceleyiniz ? ????????????????????????????????????????????????????????????????????

df.pivot_table(index=["SaleCityName", "ConceptName", "EB Score"], aggfunc=["mean","count"]).head()

df.groupby(["SaleCityName", "ConceptName", "EB Score"])["Price"].agg(["mean","count"]).head()

df.groupby(["SaleCityName", "ConceptName", "Seasons"])["Price"].agg(["mean","count"]).head()

df.groupby(["SaleCityName", "ConceptName", "CInDay"])["Price"].agg(["mean","count"]).head()

# GÖREV 4: City-Concept-Season kýrýlýmýnýn çýktýsýný PRICE'a göre sýralayýnýz. ????????????????????????????????????????????????????????

agg_df = df.pivot_table(index=["SaleCityName", "ConceptName", "Seasons"], values="Price").sort_values(by="Price", ascending=False)



# GÖREV 5: Indekste yer alan isimleri deðiþken ismine çeviriniz. ???????????????????????????????????????????????????????????????????

agg_df.reset_index(inplace=True)

# GÖREV 6: Yeni seviye tabanlý müþterileri (persona) tanýmlayýnýz  ????????????????????????????????????????????????????????????

# • Yeni seviye tabanlý satýþlarý tanýmlayýnýz ve veri setine deðiþken olarak ekleyiniz.
# • Yeni eklenecek deðiþkenin adý: sales_level_based
# • Önceki soruda elde edeceðiniz çýktýdaki gözlemleri bir araya getirerek sales_level_based deðiþkenini oluþturmanýz gerekmektedir.

agg_df["sales_level_based"] = agg_df["SaleCityName"].astype(str) + "_" + agg_df["ConceptName"].astype(str) + "_" + \
                              agg_df["Seasons"].astype(str)

# Görev 7: Yeni müþterileri (personalarý) segmentlere ayýrýnýz.

agg_df["SEGMENT"] = pd.qcut(agg_df["Price"], 4, labels=["D","C","B","A"])

# GÖREV 8: Yeni gelen müþterileri sýnýflandýrýp, ne kadar gelir getirebileceklerini tahmin ediniz.

# • Antalya’da herþey dahil ve yüksek sezonda tatil yapmak isteyen bir kiþinin ortalama ne kadar gelir kazandýrmasý beklenir?

# • Girne’de yarým pansiyon bir otele düþük sezonda giden bir tatilci hangi segmentte yer alacaktýr?

user = "Antalya_Herþey Dahil_High"

agg_df[agg_df["sales_level_based"] == user]["Price"]

user = "Girne_Yarým Pansiyon_Low"

agg_df[agg_df["sales_level_based"] == user]["Price"]