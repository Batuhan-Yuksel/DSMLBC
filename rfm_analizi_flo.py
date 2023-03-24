##########################################
# # # # # # # BATUHAN YÜKSEL # # # # # # #
##########################################

# RFM Analizi ile Müþteri Segmentasyonu

#Ýþ Problemi: Online ayakkabý maðazasý olan FLO müþterilerini segmentlere ayýrýp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.
# Buna yönelik olarak müþterilerin davranýþlarý tanýmlanacak ve bu davranýþlardaki öbeklenmelere göre gruplar oluþturulacak.

#Veri seti Flo’dan son alýþveriþlerini 2020 - 2021 yýllarýnda OmniChannel (hem online hem offline alýþveriþ yapan) olarak yapan
# müþterilerin geçmiþ alýþveriþ davranýþlarýndan elde edilen bilgilerden oluþmaktadýr.
# master_id Eþsiz müþteri numarasý
# order_channel Alýþveriþ yapýlan platforma ait hangi kanalýn kullanýldýðý (Android, ios, Desktop, Mobile)
# last_order_channel En son alýþveriþin yapýldýðý kanal
# first_order_date Müþterinin yaptýðý ilk alýþveriþ tarihi
# last_order_date Müþterinin yaptýðý son alýþveriþ tarihi
# last_order_date_online Müþterinin online platformda yaptýðý son alýþveriþ tarihi
# last_order_date_offline Müþterinin offline platformda yaptýðý son alýþveriþ tarihi
# order_num_total_ever_online Müþterinin online platformda yaptýðý toplam alýþveriþ sayýsý
# order_num_total_ever_offline Müþterinin offline'da yaptýðý toplam alýþveriþ sayýsý
# customer_value_total_ever_offline Müþterinin offline alýþveriþlerinde ödediði toplam ücret
# customer_value_total_ever_online Müþterinin online alýþveriþlerinde ödediði toplam ücret
# interested_in_categories_12 Müþterinin son 12 ayda alýþveriþ yaptýðý kategorilerin listesi


import datetime as dt
import pandas as pd

pd.set_option('display.max_columns', None)  # bütün sütunlarý gör
# pd.set_option('display.max_rows', None) # bütün satýrlarý gör
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# GÖREV 1: Veriyi Anlama ve Hazýrlama
# Adým 1: flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasýný oluþturunuz.
df_ = pd.read_csv(r"C:\Users\Batuhan\Desktop\flo_data_20k.csv")
df = df_.copy()
# Adým 2: Veri setinde
# a. Ýlk 10 gözlem,
df.head(10)
# b. Deðiþken isimleri,
df.columns
# c. Betimsel istatistik,
df.describe().T
# d. Boþ deðer,
df.isnull().values.any()
df.isnull().sum()
# e. Deðiþken tipleri, incelemesi yapýnýz.
df.info()
df.dtypes
# Adým 3: Omnichannel müþterilerin hem online'dan hemde offline platformlardan alýþveriþ yaptýðýný ifade etmektedir. Her bir müþterinin toplam
# alýþveriþ sayýsý ve harcamasý için yeni deðiþkenler oluþturunuz.

# order_num_total_ever_online Müþterinin online platformda yaptýðý toplam alýþveriþ sayýsý
# order_num_total_ever_offline Müþterinin offline'da yaptýðý toplam alýþveriþ sayýsý
df["TotalTransaction"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

# customer_value_total_ever_offline Müþterinin offline alýþveriþlerinde ödediði toplam ücret
# customer_value_total_ever_online Müþterinin online alýþveriþlerinde ödediði toplam ücret
df["TotalPrice"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]


# Adým 4: Deðiþken tiplerini inceleyiniz. Tarih ifade eden deðiþkenlerin tipini date'e çeviriniz.
df.dtypes
for col in df.columns:
    if "date" in col:
        df[col] = pd.to_datetime(df[col])

# Adým 5: Alýþveriþ kanallarýndaki müþteri sayýsýnýn, toplam alýnan ürün sayýsýnýn ve toplam harcamalarýn daðýlýmýna bakýnýz.
df.groupby("order_channel").agg({"master_id":"count","TotalTransaction":"sum","TotalPrice":"sum"})
# Adým 6: En fazla kazancý getiren ilk 10 müþteriyi sýralayýnýz.
df["TotalPrice"].sort_values(ascending=False).head(10)
# Adým 7: En fazla sipariþi veren ilk 10 müþteriyi sýralayýnýz.
df["TotalTransaction"].sort_values(ascending=False).head(10)
df[df.index == 11150]
# Adým 8: Veri ön hazýrlýk sürecini fonksiyonlaþtýrýnýz.

def prep(dataframe):
    # eksik veri kontrolü
    if dataframe.isnull().values.any() == True:
        dataframe.dropna(inplace=True)

    # iþlem ve toplam fiyatlarýn iþlenmesi
    dataframe["TotalTransaction"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["TotalPrice"] = dataframe["customer_value_total_ever_online"] + dataframe["customer_value_total_ever_offline"]

    # deðiþken tipinin deðiþtirilmesi
    for col in dataframe.columns:
        if "date" in col:
            dataframe[col] = pd.to_datetime(dataframe[col])

    return dataframe

prep(df).head()

# GÖREV 2: RFM Metriklerinin Hesaplanmasý

# Adým 1: Recency, Frequency ve Monetary tanýmlarýný yapýnýz.

# Recency Hesabý: Analizin yapýldýðý tarih - son iþlem tarihi
# Frequency: Müþterinin yaptýðý toplam iþlem sayýsý
# Monetary: Toplam iþlem tutarý yani býraktýðý toplam parasal deðer


# Adým 2: Müþteri özelinde Recency, Frequency ve Monetary metriklerini hesaplayýnýz.

df["last_order_date"].max() # en son alýþveriþin yapýldýðý tarih
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

# Adým 3: Hesapladýðýnýz metrikleri rfm isimli bir deðiþkene atayýnýz.
# Adým 2'de yapýldý.

# Adým 4: Oluþturduðunuz metriklerin isimlerini recency, frequency ve monetary olarak deðiþtiriniz.
# Adým 2'de yapýldý.

# GÖREV 3: RF Skorunun Hesaplanmasý

# Adým 1: Recency, Frequency ve Monetary metriklerini qcut yardýmý ile 1-5 arasýnda skorlara çeviriniz.
rfm["Recency_score"] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["Frequency_score"] = pd.qcut(rfm['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["Monetary_score"] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])
# Adým 2: Bu skorlarý recency_score, frequency_score ve monetary_score olarak kaydediniz.
# Adým 1'de yapýldý
# Adým 3: recency_score ve frequency_score’u tek bir deðiþken olarak ifade ediniz ve RF_SCORE olarak kaydediniz.
rfm["RF_SCORE"] = (rfm['Recency_score'].astype(str) + rfm['Frequency_score'].astype(str))
rfm["RF_SCORE"].head()

# GÖREV 4: RF Skorunun Segment Olarak Tanýmlanmasý

# Adým 1: Oluþturulan RF skorlarý için segment tanýmlamalarý yapýnýz.


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

# Adým 2: Aþaðýdaki seg_map yardýmý ile skorlarý segmentlere çeviriniz.

rfm["Segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)
rfm["Segment"].value_counts()

# GÖREV 5: Aksiyon Zamaný !

# Adým 1: Segmentlerin recency, frequnecy ve monetary ortalamalarýný inceleyiniz.
rfm[["Segment","Recency","Frequency","Monetary"]].groupby("Segment").agg("mean")
rfm.groupby("Segment")[["Recency","Frequency","Monetary"]].mean()
# Adým 2: RFM analizi yardýmýyla aþaðýda verilen 2 case için ilgili profildeki müþterileri bulun ve müþteri id'lerini csv olarak kaydediniz.
# a. FLO bünyesine yeni bir kadýn ayakkabý markasý dahil ediyor. Dahil ettiði markanýn ürün fiyatlarý genel müþteri
# tercihlerinin üstünde. Bu nedenle markanýn tanýtýmý ve ürün satýþlarý için ilgilenecek profildeki müþterilerle özel olarak
# iletiþime geçmek isteniliyor. Sadýk müþterilerinden(champions, loyal_customers) ve kadýn kategorisinden alýþveriþ
# yapan kiþiler özel olarak iletiþim kurulacak müþteriler. Bu müþterilerin id numaralarýný csv dosyasýna kaydediniz.

rfm[rfm["Segment"] == (("champions") or ("loyal_customers"))].index # segmenti champions ya da loyal_customer olan müþterilerin id bilgileri
kume1 = set(rfm[rfm["Segment"] == (("champions") or ("loyal_customers"))].index)

df[df["interested_in_categories_12"].str.contains("KADIN")]["master_id"] # 'KADIN' kategorisinden alýþveriþ yapan müþterilerin id bilgileri
kume2 = set(df[df["interested_in_categories_12"].str.contains("KADIN")]["master_id"])

new_df = pd.DataFrame(kume1.intersection(kume2), columns=["Customer_ID"])

new_df.to_csv("Customer_ID.csv")

# b. Erkek ve Çocuk ürünlerinde %40'a yakýn indirim planlanmaktadýr. Bu indirimle ilgili kategorilerle ilgilenen geçmiþte
# iyi müþteri olan ama uzun süredir alýþveriþ yapmayan kaybedilmemesi gereken müþteriler, uykuda olanlar ve yeni
# gelen müþteriler özel olarak hedef alýnmak isteniyor. Uygun profildeki müþterilerin id'lerini csv dosyasýna kaydediniz.

new_df2 = rfm[(rfm["Segment"].values == "cant_loose") |
    (rfm["Segment"].values == "hibernating") |
    (rfm["Segment"].values == "new_customers")]

new_df2.reset_index(inplace=True)

new_df2 = pd.DataFrame(new_df2["master_id"])
new_df2.to_csv("Customer_ID_2.csv")
