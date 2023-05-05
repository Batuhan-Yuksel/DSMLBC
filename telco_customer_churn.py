# G�REVLER
# � 1- Gerekli k�t�phaneleri import ediniz. Ve ard�ndan Telco Customer
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

# � 2- Telco Customer Churn veri setinin; Shape, Dtypes, Head, Tail, Eksik De�er, Describe bilgilerini elde ediniz.
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

# � 2- Veri setinin Gender s�tununda gezen ve gender s�tununun "Male" s�n�f� ile kar��la��nca 0 aksi durumla kar��la��nca 1 basan
# bir comphrension yaz�n�z ve bunu "num_gender" ad�nda yeni bir de�i�kene atay�n�z

num_gender = [0 if col == "Male" else 1 for col in df["gender"]]

# � 3- "PaperlessBilling" s�tununun s�n�flar� i�erisinde "Yes" s�n�f� i�in "Evt" , aksi durum i�in "Hyr" bast�ran bir lambda fonksiyonu yaz�n�z.
# Vesonucu "NEW_PaperlessBilling" adl� yeni olu�turdu�unuz s�tuna yazd�r�n�z. (lambda fonksiyonunu apply ile kullanabilirsiniz)

df["NEW_PaperlessBilling"] = df["PaperlessBilling"].apply(lambda x:"Evt" if x == "Yes" else "Hyr")
df["PaperlessBilling"].head()
df["NEW_PaperlessBilling"].head()

# 4- Veri setinde "Online" ifadesi i�eren s�tunlar kapsam�nda s�n�f� "Yes" olanlara "Evet", "No" olanlara "Hay�r", aksi durumda
# "�nterneti_yok" �eklinde s�n�flar� tekrar bi�imlendirecek kodu yaz�n�z.
# Not: lambda i�erisinde if elif else syntax error u ile kar��la�mamak ad�na ba�ka bir def fonksiyonu ile "yes" olanlara "Evet",
# "No" olanlara "Hay�r", aksi durumda "�nterneti_yok" �eklinde s�n�flar� tekrar bi�imlendirecek fonksiyonu d��ar�da olu�turup bu fonksiyonu
# lambda i�erisine uygulayabilirsiniz.

def check_online(col):
    if col == "Yes":
        return "Evet"
    elif col == "No":
        return "Hay�r"
    else:
        return "�nterneti_yok"

online = df.filter(like="Online")
for col in online:
    df[col] = df[col].apply(lambda x: check_online(x))




# � 5- "TotalCharges" de�i�keninin 30 dan k���k de�erlerini bulunuz. E�er hata al�rsan�z bu de�i�keninin g�zlemlerinin tipini inceleyiniz ve
# belirtilen sorgunun gelmesi i�in uygun olan tipe �evirerek sorguya devam ediniz.

df[df["TotalCharges"] < 30]
df["TotalCharges"].dtype
# df["TotalCharges"] = df["TotalCharges"].astype('float64')
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce") # �al��an bu
# df["TotalCharges"] = pd.to_numeric(df["TotalCharges"]) # ikinci yol ayn� sonu�
df["TotalCharges"].dtype
df[df["TotalCharges"] < 30]
df.head()

# � 6- �deme y�ntemi "Electronic check" olan m��terilerin ortalama Monthly Charges de�erleri ne kadard�r?

df[df["PaymentMethod"] == "Electronic check"]["MonthlyCharges"].mean()

df.groupby("PaymentMethod")["MonthlyCharges"].mean()

# � 7-Cinsiyeti kad�n olan ve internet servisi fiber optik ya da DSL olan m��terilerin toplam MonthlyCharges de�erleri ne kadard�r?

df[(df["gender"] == "Female") & ((df["InternetService"] == "Fiber optic") | (df["InternetService"] == "DSL"))]["MonthlyCharges"].sum()

# � 8- Churn de�i�keninde Yes olan s�n�flara 1 , aksi durumda 0 basan lambda fonksiyonunu Churn de�i�kenine uygulay�n�z.

df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)
df["Churn"].head()

# � 9-Veriyi Contract ve PhoneService de�i�kenerine g�re gruplay�p bu de�i�kenlerin s�n�flar�n�n Churn de�i�keninin ortalamas� ile olan
# ili�kisini inceleyiniz.

df.groupby(["Contract","PhoneService"])["Churn"].mean()

# � 10- 9. soruda istenen ��kt�n�n ayn�s�n� pivot table ile ger�ekle�tiriniz.

df.pivot_table(index="Contract", columns="PhoneService", values="Churn", aggfunc="mean")

# � 11-tenure de�i�keninin s�n�flar�n� kategorile�tirmek ad�na kendi belirledi�iniz aral�klara g�re tenure de�erlerini b�lerek
# yeni bir de�i�ken olu�turunuz. Aral�klar� labels metodu ile isimlendiriniz.

df["tenure"].head()
df["tenure"].dtype
df["tenure"].unique()
df["tenure"].nunique()
df["tenure"].min()
df["tenure"].max()

df["NEWC_tenure"] = pd.cut(df["tenure"],bins=[0,12,22,32,42,52,62,72],labels=["Cat_1","Cat_2","Cat_3","Cat_4","Cat_5","Cat_6","Cat_7"])


