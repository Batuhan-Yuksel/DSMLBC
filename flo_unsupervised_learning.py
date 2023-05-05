# GÖZETÝMSÝZ ÖÐRENME ÝLE MÜÞTERÝ SEGMENTASYONU

# ÝÞ PROBLEMÝ

# FLO müþterilerini segmentlere ayýrýp bu segmentlere göre pazarlama stratejileri belirlemek istiyor. Buna yönelik
# olarak müþterilerin davranýþlarý tanýmlanacak ve bu davranýþlardaki öbeklenmelere göre gruplar oluþturulacak.

# Veri Seti Hikayesi

# Veri seti Flo’dan son alýþveriþlerini 2020 - 2021 yýllarýnda OmniChannel (hem online hem offline alýþveriþ yapan)
# olarak yapan müþterilerin geçmiþ alýþveriþ davranýþlarýndan elde edilen bilgilerden oluþmaktadýr.


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


from warnings import filterwarnings
import warnings
filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# GÖREV 1: Veriyi Hazýrlama

# Adým 1: flo_data_20K.csv verisini okutunuz.

df_ = pd.read_csv(r"C:\Users\Batuhan\Desktop\MIUUL\3. Hafta\FLOCLTVPrediction/flo_data_20k.csv")
df = df_.copy()
df.head()
df.isnull().values.any()
df.dtypes
# Adým 2: Müþterileri segmentlerken kullanacaðýnýz deðiþkenleri seçiniz.
# Not: Tenure (Müþterinin yaþý), Recency (en son kaç gün önce alýþveriþ yaptýðý) gibi yeni deðiþkenler oluþturabilirsiniz.

for col in df.columns:
    if "date" in col:
        df[col] = pd.to_datetime(df[col])

from datetime import datetime
today_date = datetime.today().strftime('%Y-%m-%d')
today_date = pd.to_datetime(today_date)

df["Recency"] = today_date - df["last_order_date"]
df["TotalPrice"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
today_date = today_date.date()
df["first_order_date"] = pd.to_datetime(df["first_order_date"]).dt.date
df["Tenure"] = (today_date - df["first_order_date"]) / 7
df["Tenure"] = df["Tenure"].dt.days

df["TotalTransaction"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

df["Monetary"] = df["TotalPrice"] / df["TotalTransaction"]

df['Recency'] = df['Recency'].dt.days

# GÖREV 2: K-Means ile Müþteri Segmentasyonu

# Adým 1: Deðiþkenleri standartlaþtýrýnýz.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal deðiþkenlerin isimlerini verir.
    Not: Kategorik deðiþkenlerin içerisine numerik görünümlü kategorik deðiþkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Deðiþken isimleri alýnmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan deðiþkenler için sýnýf eþik deðeri
        car_th: int, optinal
                kategorik fakat kardinal deðiþkenler için sýnýf eþik deðeri

    Returns
    ------
        cat_cols: list
                Kategorik deðiþken listesi
        num_cols: list
                Numerik deðiþken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal deðiþken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam deðiþken sayýsý
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamý toplam deðiþken sayýsýna eþittir: cat_cols + num_cols + cat_but_car = deðiþken sayýsý

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=10, car_th=20)
num_cols = [col for col in num_cols if "date" not in col]

from datetime import timedelta

from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

cat_cols.append("interested_in_categories_12")

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df = one_hot_encoder(df, cat_cols)

new = [col for col in df.columns if "date" not in col]
new = [col for col in new if "master_id" not in col]

df = df[new]


# Adým 2: Optimum küme sayýsýný belirleyiniz.

from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
kmeans = KMeans(n_clusters=10, random_state=1).fit(df)

kmeans = KMeans()
ssd = []
K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklý K Deðerlerine Karþýlýk SSE/SSR/SSD")
plt.title("Optimum Küme sayýsý için Elbow Yöntemi")
plt.show()

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
elbow.show()

# Optimum küme sayýsý 8

# Adým 3: Modelinizi oluþturunuz ve müþterilerinizi segmentleyiniz.

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_

clusters_kmeans = kmeans.labels_

df = pd.read_csv(r"C:\Users\Batuhan\Desktop\MIUUL\3. Hafta\FLOCLTVPrediction/flo_data_20k.csv")

df["cluster"] = clusters_kmeans

df["cluster"] = df["cluster"] + 1

# Adým 4: Herbir segmenti istatistiksel olarak inceleyeniz.

liste = df["cluster"].unique()
liste.sort()

df["TotalPrice"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

for i in liste:
    print("Küme sýnýfý:", i)
    print("-"*120)
    print((df[df["cluster"] == i].describe().T).round(2))
    print("-"*120, "\n\n")

df.groupby("cluster")["TotalPrice"].mean().plot(kind="line")
plt.xlabel("Number of Cluster")
plt.ylabel("Total Price Mean")
plt.title("Average Sales Price by Clusters")
plt.show()



# GÖREV 3: Hierarchical Clustering ile Müþteri Segmentasyonu

# Adým 1: Görev 2'de standýrlaþtýrdýðýnýz dataframe'i kullanarak optimum küme sayýsýný belirleyiniz.

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

hc_average = linkage(df, "average")

plt.figure(figsize=(40, 20))
plt.title("Hiyerarþik Kümeleme Dendogramý")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklýklar")
dendrogram(hc_average, leaf_font_size=10)
plt.show()

# Adým 2: Modelinizi oluþturunuz ve müþterileriniz segmentleyiniz.

# küme sayýsýnýn belirlenmesi
plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average)
plt.axhline(y=0.5, color='r', linestyle='--')
plt.axhline(y=0.6, color='b', linestyle='--')
plt.show()

# final modeli
from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, linkage="average")

clusters = cluster.fit_predict(df)

df = pd.read_csv(r"C:\Users\Batuhan\Desktop\MIUUL\3. Hafta\FLOCLTVPrediction/flo_data_20k.csv")

df["hi_cluster_no"] = clusters


# Adým 3: Her bir segmenti istatistiksel olarak inceleyeniz.

for i in liste:
    print("Küme sýnýfý:", i)
    print("-"*120)
    print((df[df["hi_cluster_no"] == i].describe().T).round(2))
    print("-"*120, "\n\n")