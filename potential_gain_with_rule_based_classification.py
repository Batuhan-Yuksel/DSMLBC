import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

pd.set_option("display.width", 1000)
pd.set_option('display.max_rows', None)
pd.set_option("display.max_columns", None)

# POTENTIAL GAIN

# SaleId: Satış id
# SaleDate : Satış Tarihi
# Price: Satış için ödenen fiyat
# ConceptName:Otel konsept bilgisi
# SaleCityName: Otelin bulunduğu şehir bilgisi
# CheckInDate: Müşterinin otele giriş tarihi
# CInDay:Müşterinin otele giriş günü
# SaleCheckInDayDiff: Check in ile giriş tarihi gün farkı
# Season:Otele giriş tarihindeki sezon bilgisi

# EDA
df = pd.read_excel(r"C:\Users\Batuhan\Desktop\miuul_gezinomi.xlsx")
df.shape
df.info()
df.dtypes
df.isnull().values.any()
df.isnull().sum()
df.head()
df.describe().T

df["SaleCityName"].unique()
df["SaleCityName"].nunique()
df["SaleCityName"].value_counts()

df["ConceptName"].unique()
df["ConceptName"].nunique()

df["ConceptName"].value_counts()

df.groupby("SaleCityName")["Price"].sum()

df.groupby("ConceptName")["Price"].sum()

df.groupby("SaleCityName")["Price"].mean()

df.groupby("ConceptName")["Price"].mean()

df.pivot_table(index="ConceptName", columns="SaleCityName", values="Price", aggfunc="mean")

df["EB Score"] = pd.cut(df["SaleCheckInDayDiff"], bins=[0, 7, 30, 90, df["SaleCheckInDayDiff"].max() + 1], right=False,
                                  labels=["Last Minuters", "Potential Planners", "Planners", "Early Bookers"])


df.pivot_table(index=["SaleCityName", "ConceptName", "EB Score"], aggfunc=["mean","count"]).head()

df.groupby(["SaleCityName", "ConceptName", "EB Score"])["Price"].agg(["mean","count"]).head()

df.groupby(["SaleCityName", "ConceptName", "Seasons"])["Price"].agg(["mean","count"]).head()

df.groupby(["SaleCityName", "ConceptName", "CInDay"])["Price"].agg(["mean","count"]).head()

agg_df = df.pivot_table(index=["SaleCityName", "ConceptName", "Seasons"], values="Price").sort_values(by="Price", ascending=False)

agg_df.reset_index(inplace=True)

agg_df["sales_level_based"] = agg_df["SaleCityName"].astype(str) + "_" + agg_df["ConceptName"].astype(str) + "_" + \
                              agg_df["Seasons"].astype(str)


agg_df["SEGMENT"] = pd.qcut(agg_df["Price"], 4, labels=["D","C","B","A"])

# • Antalya’da herşey dahil ve yüksek sezonda tatil yapmak isteyen bir kişinin ortalama ne kadar gelir kazandırması beklenir?

# • Girne’de yarım pansiyon bir otele düşük sezonda giden bir tatilci hangi segmentte yer alacaktır?

user = "Antalya_Herşey Dahil_High"

agg_df[agg_df["sales_level_based"] == user]["Price"]

user = "Girne_Yarım Pansiyon_Low"

agg_df[agg_df["sales_level_based"] == user]["Price"]
