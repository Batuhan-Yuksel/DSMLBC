##########################################
# # # # # # # BATUHAN Y�KSEL # # # # # # #
##########################################
# BG-NBD ve Gamma-Gamma ile CLTV Tahmini

# �� Problemi
# FLO sat�� ve pazarlama faaliyetleri i�in roadmap belirlemek istemektedir. �irketin orta uzun vadeli plan yapabilmesi i�in
# var olan m��terilerin gelecekte �irkete sa�layacaklar� potansiyel de�erin tahmin edilmesi gerekmektedir.

# Veri Seti Hikayesi

# Veri seti Flo�dan son al��veri�lerini 2020 - 2021 y�llar�nda OmniChannel (hem online hem offline al��veri� yapan) olarak
# yapan m��terilerin ge�mi� al��veri� davran��lar�ndan elde edilen bilgilerden olu�maktad�r.

# master_id: E�siz m��teri numaras�
# order_channel: Al��veri� yap�lan platforma ait hangi kanal�n kullan�ld��� (Android, ios, Desktop, Mobile)
# last_order_channel: En son al��veri�in yap�ld��� kanal
# first_order_date: M��terinin yapt��� ilk al��veri� tarihi
# last_order_date: M��terinin yapt��� son al��veri� tarihi
# last_order_date_online: M��terinin online platformda yapt��� son al��veri� tarihi
# last_order_date_offline: M��terinin offline platformda yapt��� son al��veri� tarihi
# order_num_total_ever_online: M��terinin online platformda yapt��� toplam al��veri� say�s�
# order_num_total_ever_offline: M��terinin offline'da yapt��� toplam al��veri� say�s�
# customer_value_total_ever_offline: M��terinin offline al��veri�lerinde �dedi�i toplam �cret
# customer_value_total_ever_online: M��terinin online al��veri�lerinde �dedi�i toplam �cret
# interested_in_categories_12: M��terinin son 12 ayda al��veri� yapt��� kategorilerin listesi
# store_type: 3 farkl� companyi ifade eder. A company'sinden al��veri� yapan ki�i B'dende yapt� ise A,B �eklinde yaz�lm��t�r.

# PROJE G�REVLER�

# G�REV 1: Veriyi Haz�rlama
import datetime as dt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None) # b�t�n s�tunlar� g�ster
pd.set_option('display.width', 500) # yanyana 500 karakter g�ster
pd.set_option('display.float_format', lambda x: '%.4f' % x) # virg�lden sonra 4 basamak g�ster
from sklearn.preprocessing import MinMaxScaler

# Ad�m 1: flo_data_20K.csv verisini okuyunuz.
df_ = pd.read_csv(r"C:\Users\Batuhan\Desktop\flo_data_20k.csv")
df = df_.copy()
# Ad�m 2: Ayk�r� de�erleri bask�lamak i�in gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlar�n� tan�mlay�n�z.
# Not: cltv hesaplan�rken frequency de�erleri integer olmas� gerekmektedir.Bu nedenle alt ve �st limitlerini round() ile yuvarlay�n�z.
round(df["customer_value_total_ever_online"].quantile(0.99))
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit)

# Ad�m 3: "order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
# "customer_value_total_ever_online" de�i�kenlerinin ayk�r� de�erleri varsa bask�layan�z.
replace_with_thresholds(df,"order_num_total_ever_online")
replace_with_thresholds(df,"order_num_total_ever_offline")
replace_with_thresholds(df,"customer_value_total_ever_online")
replace_with_thresholds(df,"customer_value_total_ever_offline")
df["order_num_total_ever_online"].max()
# Ad�m 4: Omnichannel m��terilerin hem online'dan hem de offline platformlardan al��veri� yapt���n� ifade etmektedir. Her bir m��terinin toplam
# al��veri� say�s� ve harcamas� i�in yeni de�i�kenler olu�turunuz.
df["TotalTransaction"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["TotalPrice"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
# Ad�m 5: De�i�ken tiplerini inceleyiniz. Tarih ifade eden de�i�kenlerin tipini date'e �eviriniz.
df.info()
df.dtypes
for col in df.columns:
    if "date" in col:
        df[col] = pd.to_datetime(df[col])


# G�REV 2: CLTV Veri Yap�s�n�n Olu�turulmas�

# Ad�m 1: Veri setindeki en son al��veri�in yap�ld��� tarihten 2 g�n sonras�n� analiz tarihi olarak al�n�z.
df["last_order_date"].max()
today_date = dt.datetime(2021,6,1)
df["first_order_date"].min()
# Ad�m 2: customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg de�erlerinin yer ald��� yeni bir cltv dataframe'i
# olu�turunuz. Monetary de�eri sat�n alma ba��na ortalama de�er olarak, recency ve tenure de�erleri ise haftal�k cinsten ifade edilecek.


cltv_df = df.groupby("master_id").agg(recency_cltv_weekly=("last_order_date", lambda x:(today_date-x.max()).days/7),
                                   T_weekly=('first_order_date', lambda x:(today_date-x.min()).days/7),
                                   frequency=("TotalTransaction",lambda x:int(x.sum())))
cltv_df["monetary_cltv_avg"] = df["TotalPrice"].values / df["TotalTransaction"].values

cltv_df = cltv_df[cltv_df["frequency"] > 1]
# cltv = df.groupby("master_id").agg(recency_cltv_weekly=("last_order_date", lambda x:(today_date-x.max()).days),
#                                   T_weekly=('first_order_date', lambda x:(today_date-x.min()).days),
#                                   frequency=("TotalTransaction",lambda x:int(x.sum())),
#                                   monetary_cltv_avg=("TotalPrice",lambda TotalPrice:df["TotalPrice"].values / df["TotalTransaction"].values))

# G�REV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulmas� ve CLTV�nin Hesaplanmas�
# Ad�m 1: BG/NBD modelini fit ediniz.
bgf = BetaGeoFitter(penalizer_coef=0.001) # bgf ad�nda bir model nesnesi olu�turduk ve ceza katsay�s�n� 0.001 olarak belirledik
bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])
# � 3 ay i�erisinde m��terilerden beklenen sat�n almalar� tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.
cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
            cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly'])
# � 6 ay i�erisinde m��terilerden beklenen sat�n almalar� tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.
cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
            cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly'])

# bg-nbd sat�n alma/i�lem say�s�n� modeller, gamma gamma ise average profit de�erini modeller

# Ad�m 2: Gamma-Gamma modelini fit ediniz. M��terilerin ortalama b�rakacaklar� de�eri tahminleyip exp_average_value olarak cltv
# dataframe'ine ekleyiniz.
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg']) # model kurduk
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
# Ad�m 3: 6 ayl�k CLTV hesaplay�n�z ve cltv ismiyle dataframe'e ekleyiniz.
cltv_df["cltv"] = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,  # 6 ayl�k.
                                   freq="W")  # T'nin frekans bilgisi. verinin frekans bilgisi. bizim olu�turdu�umuz veri haftal�k o y�zden W
# � Cltv de�eri en y�ksek 20 ki�iyi g�zlemleyiniz.
cltv_df["cltv"].sort_values(ascending=False).head(20)

# G�REV 4: CLTV De�erine G�re Segmentlerin Olu�turulmas�
# Ad�m 1: 6 ayl�k CLTV'ye g�re t�m m��terilerinizi 4 gruba (segmente) ay�r�n�z ve grup isimlerini veri setine ekleyiniz.
cltv_df["Segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])
# Ad�m 2: 4 grup i�erisinden se�ece�iniz 2 grup i�in y�netime k�sa k�sa 6 ayl�k aksiyon �nerilerinde bulununuz.
cltv_df.groupby("Segment")["cltv"].mean()