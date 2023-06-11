##########################################
# # # # # # # BATUHAN YÜKSEL # # # # # # #
##########################################

#Customer Segmentation with RFM Analysis

# Veri seti Flo’dan son alışverişlerini 2020 - 2021 yıllarında OmniChannel (hem online hem offline alışveriş yapan) olarak yapan
# müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.
# master_id Eşsiz müşteri numarası
# order_channel Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
# last_order_channel En son alışverişin yapıldığı kanal
# first_order_date Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online Müşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline Müşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi


import datetime as dt
import pandas as pd

pd.set_option('display.max_columns', None)  # bütün sütunları gör
# pd.set_option('display.max_rows', None) # bütün satırları gör
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_ = pd.read_csv(r"C:\Users\Batuhan\Desktop\flo_data_20k.csv")
df = df_.copy()
df.columns
df.describe().T
df.isnull().values.any()
df.isnull().sum()
df.info()
df.dtypes

df["TotalTransaction"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

df["TotalPrice"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

df.dtypes
for col in df.columns:
    if "date" in col:
        df[col] = pd.to_datetime(df[col])

df.groupby("order_channel").agg({"master_id":"count","TotalTransaction":"sum","TotalPrice":"sum"})

df["TotalPrice"].sort_values(ascending=False).head(10)

df["TotalTransaction"].sort_values(ascending=False).head(10)
df[df.index == 11150]


def prep(dataframe):
    
    if dataframe.isnull().values.any() == True:
        dataframe.dropna(inplace=True)

    
    dataframe["TotalTransaction"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["TotalPrice"] = dataframe["customer_value_total_ever_online"] + dataframe["customer_value_total_ever_offline"]

    
    for col in dataframe.columns:
        if "date" in col:
            dataframe[col] = pd.to_datetime(dataframe[col])

    return dataframe

prep(df).head()

# Recency Hesabı: Analizin yapıldığı tarih - son işlem tarihi
# Frequency: Müşterinin yaptığı toplam işlem sayısı
# Monetary: Toplam işlem tutarı yani bıraktığı toplam parasal değer

df["last_order_date"].max()
type(df["last_order_date"])
today_date = dt.datetime(2021, 6, 1)
type(today_date)

# df.groupby("master_id").agg({"last_order_date":lambda last_order_date: (today_date - last_order_date.max()).days,
#                             "TotalTransaction":lambda TotalTransaction: TotalTransaction.sum(),
#                             "TotalPrice":lambda TotalPrice: TotalPrice.sum()})

rfm = df.groupby("master_id").agg(Recency = ("last_order_date", lambda last_order_date: (today_date - last_order_date.max()).days),
                            Frequency = ("TotalTransaction", lambda TotalTransaction: int(TotalTransaction.sum())),
                            Monetary = ("TotalPrice", lambda TotalPrice: TotalPrice.sum()))

# df.groupby("master_id").agg(
#    Recency = pd.NamedAgg(column="last_order_date",aggfunc=lambda last_order_date: (today_date - last_order_date.max()).days),
#    Frequency = pd.NamedAgg(column="TotalTransasction",aggfunc=lambda TotalTransaction: int(TotalTransaction.sum())),
#    Monetary = pd.NamedAgg(column="TotalPrice",aggfunc=lambda TotalPrice: TotalPrice.sum())
# )

rfm["Recency_score"] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["Frequency_score"] = pd.qcut(rfm['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["Monetary_score"] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm["RF_SCORE"] = (rfm['Recency_score'].astype(str) + rfm['Frequency_score'].astype(str))
rfm["RF_SCORE"].head()

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm["Segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)
rfm["Segment"].value_counts()


rfm[["Segment","Recency","Frequency","Monetary"]].groupby("Segment").agg("mean")
rfm.groupby("Segment")[["Recency","Frequency","Monetary"]].mean()

# FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri
# tercihlerinin üstünde. Bu nedenle markanın tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak
# iletişime geçmek isteniliyor. Sadık müşterilerinden(champions, loyal_customers) ve kadın kategorisinden alışveriş
# yapan kişiler özel olarak iletişim kurulacak müşteriler. Bu müşterilerin id numaraları

rfm[rfm["Segment"] == (("champions") or ("loyal_customers"))].index # segmenti champions ya da loyal_customer olan müşterilerin id bilgileri
kume1 = set(rfm[rfm["Segment"] == (("champions") or ("loyal_customers"))].index)

df[df["interested_in_categories_12"].str.contains("KADIN")]["master_id"] # 'KADIN' kategorisinden alışveriş yapan müşterilerin id bilgileri
kume2 = set(df[df["interested_in_categories_12"].str.contains("KADIN")]["master_id"])

new_df = pd.DataFrame(kume1.intersection(kume2), columns=["Customer_ID"])

new_df.to_csv("Customer_ID.csv")

# Erkek ve Çocuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte
# iyi müşteri olan ama uzun süredir alışveriş yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve yeni
# gelen müşteriler özel olarak hedef alınmak isteniyor. Uygun profildeki müşterilerin id'leri

new_df2 = rfm[(rfm["Segment"].values == "cant_loose") |
    (rfm["Segment"].values == "hibernating") |
    (rfm["Segment"].values == "new_customers")]

new_df2.reset_index(inplace=True)

new_df2 = pd.DataFrame(new_df2["master_id"])
new_df2.to_csv("Customer_ID_2.csv")
