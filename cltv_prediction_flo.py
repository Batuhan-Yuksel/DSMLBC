##########################################
# # # # # # # BATUHAN YÜKSEL # # # # # # #
##########################################
# BG-NBD ve Gamma-Gamma ile CLTV Tahmini

# Ýþ Problemi
# FLO satýþ ve pazarlama faaliyetleri için roadmap belirlemek istemektedir. Þirketin orta uzun vadeli plan yapabilmesi için
# var olan müþterilerin gelecekte þirkete saðlayacaklarý potansiyel deðerin tahmin edilmesi gerekmektedir.

# Veri Seti Hikayesi

# Veri seti Flo’dan son alýþveriþlerini 2020 - 2021 yýllarýnda OmniChannel (hem online hem offline alýþveriþ yapan) olarak
# yapan müþterilerin geçmiþ alýþveriþ davranýþlarýndan elde edilen bilgilerden oluþmaktadýr.

# master_id: Eþsiz müþteri numarasý
# order_channel: Alýþveriþ yapýlan platforma ait hangi kanalýn kullanýldýðý (Android, ios, Desktop, Mobile)
# last_order_channel: En son alýþveriþin yapýldýðý kanal
# first_order_date: Müþterinin yaptýðý ilk alýþveriþ tarihi
# last_order_date: Müþterinin yaptýðý son alýþveriþ tarihi
# last_order_date_online: Müþterinin online platformda yaptýðý son alýþveriþ tarihi
# last_order_date_offline: Müþterinin offline platformda yaptýðý son alýþveriþ tarihi
# order_num_total_ever_online: Müþterinin online platformda yaptýðý toplam alýþveriþ sayýsý
# order_num_total_ever_offline: Müþterinin offline'da yaptýðý toplam alýþveriþ sayýsý
# customer_value_total_ever_offline: Müþterinin offline alýþveriþlerinde ödediði toplam ücret
# customer_value_total_ever_online: Müþterinin online alýþveriþlerinde ödediði toplam ücret
# interested_in_categories_12: Müþterinin son 12 ayda alýþveriþ yaptýðý kategorilerin listesi
# store_type: 3 farklý companyi ifade eder. A company'sinden alýþveriþ yapan kiþi B'dende yaptý ise A,B þeklinde yazýlmýþtýr.

# PROJE GÖREVLERÝ

# GÖREV 1: Veriyi Hazýrlama
import datetime as dt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None) # bütün sütunlarý göster
pd.set_option('display.width', 500) # yanyana 500 karakter göster
pd.set_option('display.float_format', lambda x: '%.4f' % x) # virgülden sonra 4 basamak göster
from sklearn.preprocessing import MinMaxScaler

# Adým 1: flo_data_20K.csv verisini okuyunuz.
df_ = pd.read_csv(r"C:\Users\Batuhan\Desktop\flo_data_20k.csv")
df = df_.copy()
# Adým 2: Aykýrý deðerleri baskýlamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarýný tanýmlayýnýz.
# Not: cltv hesaplanýrken frequency deðerleri integer olmasý gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayýnýz.
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

# Adým 3: "order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
# "customer_value_total_ever_online" deðiþkenlerinin aykýrý deðerleri varsa baskýlayanýz.
replace_with_thresholds(df,"order_num_total_ever_online")
replace_with_thresholds(df,"order_num_total_ever_offline")
replace_with_thresholds(df,"customer_value_total_ever_online")
replace_with_thresholds(df,"customer_value_total_ever_offline")
df["order_num_total_ever_online"].max()
# Adým 4: Omnichannel müþterilerin hem online'dan hem de offline platformlardan alýþveriþ yaptýðýný ifade etmektedir. Her bir müþterinin toplam
# alýþveriþ sayýsý ve harcamasý için yeni deðiþkenler oluþturunuz.
df["TotalTransaction"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["TotalPrice"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
# Adým 5: Deðiþken tiplerini inceleyiniz. Tarih ifade eden deðiþkenlerin tipini date'e çeviriniz.
df.info()
df.dtypes
for col in df.columns:
    if "date" in col:
        df[col] = pd.to_datetime(df[col])


# GÖREV 2: CLTV Veri Yapýsýnýn Oluþturulmasý

# Adým 1: Veri setindeki en son alýþveriþin yapýldýðý tarihten 2 gün sonrasýný analiz tarihi olarak alýnýz.
df["last_order_date"].max()
today_date = dt.datetime(2021,6,1)
df["first_order_date"].min()
# Adým 2: customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg deðerlerinin yer aldýðý yeni bir cltv dataframe'i
# oluþturunuz. Monetary deðeri satýn alma baþýna ortalama deðer olarak, recency ve tenure deðerleri ise haftalýk cinsten ifade edilecek.


cltv_df = df.groupby("master_id").agg(recency_cltv_weekly=("last_order_date", lambda x:(today_date-x.max()).days/7),
                                   T_weekly=('first_order_date', lambda x:(today_date-x.min()).days/7),
                                   frequency=("TotalTransaction",lambda x:int(x.sum())))
cltv_df["monetary_cltv_avg"] = df["TotalPrice"].values / df["TotalTransaction"].values

cltv_df = cltv_df[cltv_df["frequency"] > 1]
# cltv = df.groupby("master_id").agg(recency_cltv_weekly=("last_order_date", lambda x:(today_date-x.max()).days),
#                                   T_weekly=('first_order_date', lambda x:(today_date-x.min()).days),
#                                   frequency=("TotalTransaction",lambda x:int(x.sum())),
#                                   monetary_cltv_avg=("TotalPrice",lambda TotalPrice:df["TotalPrice"].values / df["TotalTransaction"].values))

# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulmasý ve CLTV’nin Hesaplanmasý
# Adým 1: BG/NBD modelini fit ediniz.
bgf = BetaGeoFitter(penalizer_coef=0.001) # bgf adýnda bir model nesnesi oluþturduk ve ceza katsayýsýný 0.001 olarak belirledik
bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])
# • 3 ay içerisinde müþterilerden beklenen satýn almalarý tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.
cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
            cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly'])
# • 6 ay içerisinde müþterilerden beklenen satýn almalarý tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.
cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
            cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly'])

# bg-nbd satýn alma/iþlem sayýsýný modeller, gamma gamma ise average profit deðerini modeller

# Adým 2: Gamma-Gamma modelini fit ediniz. Müþterilerin ortalama býrakacaklarý deðeri tahminleyip exp_average_value olarak cltv
# dataframe'ine ekleyiniz.
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg']) # model kurduk
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
# Adým 3: 6 aylýk CLTV hesaplayýnýz ve cltv ismiyle dataframe'e ekleyiniz.
cltv_df["cltv"] = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,  # 6 aylýk.
                                   freq="W")  # T'nin frekans bilgisi. verinin frekans bilgisi. bizim oluþturduðumuz veri haftalýk o yüzden W
# • Cltv deðeri en yüksek 20 kiþiyi gözlemleyiniz.
cltv_df["cltv"].sort_values(ascending=False).head(20)

# GÖREV 4: CLTV Deðerine Göre Segmentlerin Oluþturulmasý
# Adým 1: 6 aylýk CLTV'ye göre tüm müþterilerinizi 4 gruba (segmente) ayýrýnýz ve grup isimlerini veri setine ekleyiniz.
cltv_df["Segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])
# Adým 2: 4 grup içerisinden seçeceðiniz 2 grup için yönetime kýsa kýsa 6 aylýk aksiyon önerilerinde bulununuz.
cltv_df.groupby("Segment")["cltv"].mean()