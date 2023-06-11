##########################################
# # # # # # # BATUHAN YÜKSEL # # # # # # #
##########################################
# BG-NBD ve Gamma-Gamma ile CLTV Tahmini

# İş Problemi
# FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir. Şirketin orta uzun vadeli plan yapabilmesi için
# var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.

# Veri seti Flo’dan son alışverişlerini 2020 - 2021 yıllarında OmniChannel (hem online hem offline alışveriş yapan) olarak
# yapan müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel: Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
# last_order_channel: En son alışverişin yapıldığı kanal
# first_order_date: Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date: Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online: Müşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline: Müşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online: Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline: Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline: Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online: Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12: Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi
# store_type: 3 farklı companyi ifade eder. A company'sinden alışveriş yapan kişi B'dende yaptı ise A,B şeklinde yazılmıştır.

import datetime as dt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500) 
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler

df_ = pd.read_csv(r"C:\Users\Batuhan\Desktop\flo_data_20k.csv")
df = df_.copy()

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

replace_with_thresholds(df,"order_num_total_ever_online")
replace_with_thresholds(df,"order_num_total_ever_offline")
replace_with_thresholds(df,"customer_value_total_ever_online")
replace_with_thresholds(df,"customer_value_total_ever_offline")
df["order_num_total_ever_online"].max()

df["TotalTransaction"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["TotalPrice"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

df.info()
df.dtypes
for col in df.columns:
    if "date" in col:
        df[col] = pd.to_datetime(df[col])

df["last_order_date"].max()
today_date = dt.datetime(2021,6,1)
df["first_order_date"].min()

cltv_df = df.groupby("master_id").agg(recency_cltv_weekly=("last_order_date", lambda x:(today_date-x.max()).days/7),
                                   T_weekly=('first_order_date', lambda x:(today_date-x.min()).days/7),
                                   frequency=("TotalTransaction",lambda x:int(x.sum())))
cltv_df["monetary_cltv_avg"] = df["TotalPrice"].values / df["TotalTransaction"].values

cltv_df = cltv_df[cltv_df["frequency"] > 1]
# cltv = df.groupby("master_id").agg(recency_cltv_weekly=("last_order_date", lambda x:(today_date-x.max()).days),
#                                   T_weekly=('first_order_date', lambda x:(today_date-x.min()).days),
#                                   frequency=("TotalTransaction",lambda x:int(x.sum())),
#                                   monetary_cltv_avg=("TotalPrice",lambda TotalPrice:df["TotalPrice"].values / df["TotalTransaction"].values))

# BG/NBD, Gamma-Gamma Modellerinin Kurulması ve CLTV’nin Hesaplanması
bgf = BetaGeoFitter(penalizer_coef=0.001) # bgf adında bir model nesnesi oluşturduk ve ceza katsayısını 0.001 olarak belirledik
bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])
# • 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.
cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
            cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly'])
# • 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.
cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
            cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly'])

# bg-nbd satın alma/işlem sayısını modeller, gamma gamma ise average profit değerini modeller

ggf = GammaGammaFitter(penalizer_coef=0.01)
# Gamma-Gamma Modelling
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
# Adım 3: 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
cltv_df["cltv"] = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,  # 6 aylık.
                                   freq="W")  # T'nin frekans bilgisi. verinin frekans bilgisi. bizim oluşturduğumuz veri haftalık o yüzden W

cltv_df["cltv"].sort_values(ascending=False).head(20)

cltv_df["Segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])
