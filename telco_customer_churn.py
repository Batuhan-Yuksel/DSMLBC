# GÖREVLER
# • 1- Gerekli kütüphaneleri import ediniz. Ve ardýndan Telco Customer
# Churn veri setini okutunuz.

import numpy as np
import pandas as pd
import seaborn as sns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
telco = pd.read_csv(r"C:\Users\Batuhan\Desktop\Telco-Customer-Churn.csv")
# telco = pd.read_csv("C:\\Users\Batuhan\\Desktop\\Telco-Customer-Churn.csv") # iki versiyonu da okudu
df = telco.copy()

# • 2- Telco Customer Churn veri setinin; Shape, Dtypes, Head, Tail, Eksik Deðer, Describe bilgilerini elde ediniz.
df.shape
df.dtypes
df.head()
df.tail()
df.isnull().values.any()
df.isnull().sum()
df.isnull().sum().sum()
df.describe().T
for col in df.columns:
    if df[col].dtype == ("float64") or df[col].dtype == ("int64"):
        print(col)
df["SeniorCitizen"].unique()

# • 2- Veri setinin Gender sütununda gezen ve gender sütununun "Male" sýnýfý ile karþýlaþýnca 0 aksi durumla karþýlaþýnca 1 basan
# bir comphrension yazýnýz ve bunu "num_gender" adýnda yeni bir deðiþkene atayýnýz

num_gender = [0 if col == "Male" else 1 for col in df["gender"]]

# • 3- "PaperlessBilling" sütununun sýnýflarý içerisinde "Yes" sýnýfý için "Evt" , aksi durum için "Hyr" bastýran bir lambda fonksiyonu yazýnýz.
# Vesonucu "NEW_PaperlessBilling" adlý yeni oluþturduðunuz sütuna yazdýrýnýz. (lambda fonksiyonunu apply ile kullanabilirsiniz)

df["NEW_PaperlessBilling"] = df["PaperlessBilling"].apply(lambda x:"Evt" if x == "Yes" else "Hyr")
df["PaperlessBilling"].head()
df["NEW_PaperlessBilling"].head()

# 4- Veri setinde "Online" ifadesi içeren sütunlar kapsamýnda sýnýfý "Yes" olanlara "Evet", "No" olanlara "Hayýr", aksi durumda
# "Ýnterneti_yok" þeklinde sýnýflarý tekrar biçimlendirecek kodu yazýnýz.
# Not: lambda içerisinde if elif else syntax error u ile karþýlaþmamak adýna baþka bir def fonksiyonu ile "yes" olanlara "Evet",
# "No" olanlara "Hayýr", aksi durumda "Ýnterneti_yok" þeklinde sýnýflarý tekrar biçimlendirecek fonksiyonu dýþarýda oluþturup bu fonksiyonu
# lambda içerisine uygulayabilirsiniz.

def check_online(col):
    if col == "Yes":
        return "Evet"
    elif col == "No":
        return "Hayýr"
    else:
        return "Ýnterneti_yok"

online = df.filter(like="Online")
for col in online:
    df[col] = df[col].apply(lambda x: check_online(x))




# • 5- "TotalCharges" deðiþkeninin 30 dan küçük deðerlerini bulunuz. Eðer hata alýrsanýz bu deðiþkeninin gözlemlerinin tipini inceleyiniz ve
# belirtilen sorgunun gelmesi için uygun olan tipe çevirerek sorguya devam ediniz.

df[df["TotalCharges"] < 30]
df["TotalCharges"].dtype
# df["TotalCharges"] = df["TotalCharges"].astype('float64')
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce") # çalýþan bu
# df["TotalCharges"] = pd.to_numeric(df["TotalCharges"]) # ikinci yol ayný sonuç
df["TotalCharges"].dtype
df[df["TotalCharges"] < 30]
df.head()

# • 6- Ödeme yöntemi "Electronic check" olan müþterilerin ortalama Monthly Charges deðerleri ne kadardýr?

df[df["PaymentMethod"] == "Electronic check"]["MonthlyCharges"].mean()

df.groupby("PaymentMethod")["MonthlyCharges"].mean()

# • 7-Cinsiyeti kadýn olan ve internet servisi fiber optik ya da DSL olan müþterilerin toplam MonthlyCharges deðerleri ne kadardýr?

df[(df["gender"] == "Female") & ((df["InternetService"] == "Fiber optic") | (df["InternetService"] == "DSL"))]["MonthlyCharges"].sum()

# • 8- Churn deðiþkeninde Yes olan sýnýflara 1 , aksi durumda 0 basan lambda fonksiyonunu Churn deðiþkenine uygulayýnýz.

df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)
df["Churn"].head()

# • 9-Veriyi Contract ve PhoneService deðiþkenerine göre gruplayýp bu deðiþkenlerin sýnýflarýnýn Churn deðiþkeninin ortalamasý ile olan
# iliþkisini inceleyiniz.

df.groupby(["Contract","PhoneService"])["Churn"].mean()

# • 10- 9. soruda istenen çýktýnýn aynýsýný pivot table ile gerçekleþtiriniz.

df.pivot_table(index="Contract", columns="PhoneService", values="Churn", aggfunc="mean")

# • 11-tenure deðiþkeninin sýnýflarýný kategorileþtirmek adýna kendi belirlediðiniz aralýklara göre tenure deðerlerini bölerek
# yeni bir deðiþken oluþturunuz. Aralýklarý labels metodu ile isimlendiriniz.

df["tenure"].head()
df["tenure"].dtype
df["tenure"].unique()
df["tenure"].nunique()
df["tenure"].min()
df["tenure"].max()

df["NEWC_tenure"] = pd.cut(df["tenure"],bins=[0,12,22,32,42,52,62,72],labels=["Cat_1","Cat_2","Cat_3","Cat_4","Cat_5","Cat_6","Cat_7"])


