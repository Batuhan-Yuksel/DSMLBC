##########################################
# # # # # # # BATUHAN Y�KSEL # # # # # # #
##########################################

# RFM Analizi ile M��teri Segmentasyonu

#�� Problemi: Online ayakkab� ma�azas� olan FLO m��terilerini segmentlere ay�r�p bu segmentlere g�re pazarlama stratejileri belirlemek istiyor.
# Buna y�nelik olarak m��terilerin davran��lar� tan�mlanacak ve bu davran��lardaki �beklenmelere g�re gruplar olu�turulacak.

#Veri seti Flo�dan son al��veri�lerini 2020 - 2021 y�llar�nda OmniChannel (hem online hem offline al��veri� yapan) olarak yapan
# m��terilerin ge�mi� al��veri� davran��lar�ndan elde edilen bilgilerden olu�maktad�r.
# master_id E�siz m��teri numaras�
# order_channel Al��veri� yap�lan platforma ait hangi kanal�n kullan�ld��� (Android, ios, Desktop, Mobile)
# last_order_channel En son al��veri�in yap�ld��� kanal
# first_order_date M��terinin yapt��� ilk al��veri� tarihi
# last_order_date M��terinin yapt��� son al��veri� tarihi
# last_order_date_online M��terinin online platformda yapt��� son al��veri� tarihi
# last_order_date_offline M��terinin offline platformda yapt��� son al��veri� tarihi
# order_num_total_ever_online M��terinin online platformda yapt��� toplam al��veri� say�s�
# order_num_total_ever_offline M��terinin offline'da yapt��� toplam al��veri� say�s�
# customer_value_total_ever_offline M��terinin offline al��veri�lerinde �dedi�i toplam �cret
# customer_value_total_ever_online M��terinin online al��veri�lerinde �dedi�i toplam �cret
# interested_in_categories_12 M��terinin son 12 ayda al��veri� yapt��� kategorilerin listesi


import datetime as dt
import pandas as pd

pd.set_option('display.max_columns', None)  # b�t�n s�tunlar� g�r
# pd.set_option('display.max_rows', None) # b�t�n sat�rlar� g�r
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# G�REV 1: Veriyi Anlama ve Haz�rlama
# Ad�m 1: flo_data_20K.csv verisini okuyunuz.Dataframe�in kopyas�n� olu�turunuz.
df_ = pd.read_csv(r"C:\Users\Batuhan\Desktop\flo_data_20k.csv")
df = df_.copy()
# Ad�m 2: Veri setinde
# a. �lk 10 g�zlem,
df.head(10)
# b. De�i�ken isimleri,
df.columns
# c. Betimsel istatistik,
df.describe().T
# d. Bo� de�er,
df.isnull().values.any()
df.isnull().sum()
# e. De�i�ken tipleri, incelemesi yap�n�z.
df.info()
df.dtypes
# Ad�m 3: Omnichannel m��terilerin hem online'dan hemde offline platformlardan al��veri� yapt���n� ifade etmektedir. Her bir m��terinin toplam
# al��veri� say�s� ve harcamas� i�in yeni de�i�kenler olu�turunuz.

# order_num_total_ever_online M��terinin online platformda yapt��� toplam al��veri� say�s�
# order_num_total_ever_offline M��terinin offline'da yapt��� toplam al��veri� say�s�
df["TotalTransaction"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

# customer_value_total_ever_offline M��terinin offline al��veri�lerinde �dedi�i toplam �cret
# customer_value_total_ever_online M��terinin online al��veri�lerinde �dedi�i toplam �cret
df["TotalPrice"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]


# Ad�m 4: De�i�ken tiplerini inceleyiniz. Tarih ifade eden de�i�kenlerin tipini date'e �eviriniz.
df.dtypes
for col in df.columns:
    if "date" in col:
        df[col] = pd.to_datetime(df[col])

# Ad�m 5: Al��veri� kanallar�ndaki m��teri say�s�n�n, toplam al�nan �r�n say�s�n�n ve toplam harcamalar�n da��l�m�na bak�n�z.
df.groupby("order_channel").agg({"master_id":"count","TotalTransaction":"sum","TotalPrice":"sum"})
# Ad�m 6: En fazla kazanc� getiren ilk 10 m��teriyi s�ralay�n�z.
df["TotalPrice"].sort_values(ascending=False).head(10)
# Ad�m 7: En fazla sipari�i veren ilk 10 m��teriyi s�ralay�n�z.
df["TotalTransaction"].sort_values(ascending=False).head(10)
df[df.index == 11150]
# Ad�m 8: Veri �n haz�rl�k s�recini fonksiyonla�t�r�n�z.

def prep(dataframe):
    # eksik veri kontrol�
    if dataframe.isnull().values.any() == True:
        dataframe.dropna(inplace=True)

    # i�lem ve toplam fiyatlar�n i�lenmesi
    dataframe["TotalTransaction"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["TotalPrice"] = dataframe["customer_value_total_ever_online"] + dataframe["customer_value_total_ever_offline"]

    # de�i�ken tipinin de�i�tirilmesi
    for col in dataframe.columns:
        if "date" in col:
            dataframe[col] = pd.to_datetime(dataframe[col])

    return dataframe

prep(df).head()

# G�REV 2: RFM Metriklerinin Hesaplanmas�

# Ad�m 1: Recency, Frequency ve Monetary tan�mlar�n� yap�n�z.

# Recency Hesab�: Analizin yap�ld��� tarih - son i�lem tarihi
# Frequency: M��terinin yapt��� toplam i�lem say�s�
# Monetary: Toplam i�lem tutar� yani b�rakt��� toplam parasal de�er


# Ad�m 2: M��teri �zelinde Recency, Frequency ve Monetary metriklerini hesaplay�n�z.

df["last_order_date"].max() # en son al��veri�in yap�ld��� tarih
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

# Ad�m 3: Hesaplad���n�z metrikleri rfm isimli bir de�i�kene atay�n�z.
# Ad�m 2'de yap�ld�.

# Ad�m 4: Olu�turdu�unuz metriklerin isimlerini recency, frequency ve monetary olarak de�i�tiriniz.
# Ad�m 2'de yap�ld�.

# G�REV 3: RF Skorunun Hesaplanmas�

# Ad�m 1: Recency, Frequency ve Monetary metriklerini qcut yard�m� ile 1-5 aras�nda skorlara �eviriniz.
rfm["Recency_score"] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["Frequency_score"] = pd.qcut(rfm['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["Monetary_score"] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])
# Ad�m 2: Bu skorlar� recency_score, frequency_score ve monetary_score olarak kaydediniz.
# Ad�m 1'de yap�ld�
# Ad�m 3: recency_score ve frequency_score�u tek bir de�i�ken olarak ifade ediniz ve RF_SCORE olarak kaydediniz.
rfm["RF_SCORE"] = (rfm['Recency_score'].astype(str) + rfm['Frequency_score'].astype(str))
rfm["RF_SCORE"].head()

# G�REV 4: RF Skorunun Segment Olarak Tan�mlanmas�

# Ad�m 1: Olu�turulan RF skorlar� i�in segment tan�mlamalar� yap�n�z.


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

# Ad�m 2: A�a��daki seg_map yard�m� ile skorlar� segmentlere �eviriniz.

rfm["Segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)
rfm["Segment"].value_counts()

# G�REV 5: Aksiyon Zaman� !

# Ad�m 1: Segmentlerin recency, frequnecy ve monetary ortalamalar�n� inceleyiniz.
rfm[["Segment","Recency","Frequency","Monetary"]].groupby("Segment").agg("mean")
rfm.groupby("Segment")[["Recency","Frequency","Monetary"]].mean()
# Ad�m 2: RFM analizi yard�m�yla a�a��da verilen 2 case i�in ilgili profildeki m��terileri bulun ve m��teri id'lerini csv olarak kaydediniz.
# a. FLO b�nyesine yeni bir kad�n ayakkab� markas� dahil ediyor. Dahil etti�i markan�n �r�n fiyatlar� genel m��teri
# tercihlerinin �st�nde. Bu nedenle markan�n tan�t�m� ve �r�n sat��lar� i�in ilgilenecek profildeki m��terilerle �zel olarak
# ileti�ime ge�mek isteniliyor. Sad�k m��terilerinden(champions, loyal_customers) ve kad�n kategorisinden al��veri�
# yapan ki�iler �zel olarak ileti�im kurulacak m��teriler. Bu m��terilerin id numaralar�n� csv dosyas�na kaydediniz.

rfm[rfm["Segment"] == (("champions") or ("loyal_customers"))].index # segmenti champions ya da loyal_customer olan m��terilerin id bilgileri
kume1 = set(rfm[rfm["Segment"] == (("champions") or ("loyal_customers"))].index)

df[df["interested_in_categories_12"].str.contains("KADIN")]["master_id"] # 'KADIN' kategorisinden al��veri� yapan m��terilerin id bilgileri
kume2 = set(df[df["interested_in_categories_12"].str.contains("KADIN")]["master_id"])

new_df = pd.DataFrame(kume1.intersection(kume2), columns=["Customer_ID"])

new_df.to_csv("Customer_ID.csv")

# b. Erkek ve �ocuk �r�nlerinde %40'a yak�n indirim planlanmaktad�r. Bu indirimle ilgili kategorilerle ilgilenen ge�mi�te
# iyi m��teri olan ama uzun s�redir al��veri� yapmayan kaybedilmemesi gereken m��teriler, uykuda olanlar ve yeni
# gelen m��teriler �zel olarak hedef al�nmak isteniyor. Uygun profildeki m��terilerin id'lerini csv dosyas�na kaydediniz.

new_df2 = rfm[(rfm["Segment"].values == "cant_loose") |
    (rfm["Segment"].values == "hibernating") |
    (rfm["Segment"].values == "new_customers")]

new_df2.reset_index(inplace=True)

new_df2 = pd.DataFrame(new_df2["master_id"])
new_df2.to_csv("Customer_ID_2.csv")
