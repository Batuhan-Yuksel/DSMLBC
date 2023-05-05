# G�ZET�MS�Z ��RENME �LE M��TER� SEGMENTASYONU

# �� PROBLEM�

# FLO m��terilerini segmentlere ay�r�p bu segmentlere g�re pazarlama stratejileri belirlemek istiyor. Buna y�nelik
# olarak m��terilerin davran��lar� tan�mlanacak ve bu davran��lardaki �beklenmelere g�re gruplar olu�turulacak.

# Veri Seti Hikayesi

# Veri seti Flo�dan son al��veri�lerini 2020 - 2021 y�llar�nda OmniChannel (hem online hem offline al��veri� yapan)
# olarak yapan m��terilerin ge�mi� al��veri� davran��lar�ndan elde edilen bilgilerden olu�maktad�r.


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

# G�REV 1: Veriyi Haz�rlama

# Ad�m 1: flo_data_20K.csv verisini okutunuz.

df_ = pd.read_csv(r"C:\Users\Batuhan\Desktop\MIUUL\3. Hafta\FLOCLTVPrediction/flo_data_20k.csv")
df = df_.copy()
df.head()
df.isnull().values.any()
df.dtypes
# Ad�m 2: M��terileri segmentlerken kullanaca��n�z de�i�kenleri se�iniz.
# Not: Tenure (M��terinin ya��), Recency (en son ka� g�n �nce al��veri� yapt���) gibi yeni de�i�kenler olu�turabilirsiniz.

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

# G�REV 2: K-Means ile M��teri Segmentasyonu

# Ad�m 1: De�i�kenleri standartla�t�r�n�z.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal de�i�kenlerin isimlerini verir.
    Not: Kategorik de�i�kenlerin i�erisine numerik g�r�n�ml� kategorik de�i�kenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                De�i�ken isimleri al�nmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan de�i�kenler i�in s�n�f e�ik de�eri
        car_th: int, optinal
                kategorik fakat kardinal de�i�kenler i�in s�n�f e�ik de�eri

    Returns
    ------
        cat_cols: list
                Kategorik de�i�ken listesi
        num_cols: list
                Numerik de�i�ken listesi
        cat_but_car: list
                Kategorik g�r�n�ml� kardinal de�i�ken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam de�i�ken say�s�
        num_but_cat cat_cols'un i�erisinde.
        Return olan 3 liste toplam� toplam de�i�ken say�s�na e�ittir: cat_cols + num_cols + cat_but_car = de�i�ken say�s�

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


# Ad�m 2: Optimum k�me say�s�n� belirleyiniz.

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
plt.xlabel("Farkl� K De�erlerine Kar��l�k SSE/SSR/SSD")
plt.title("Optimum K�me say�s� i�in Elbow Y�ntemi")
plt.show()

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
elbow.show()

# Optimum k�me say�s� 8

# Ad�m 3: Modelinizi olu�turunuz ve m��terilerinizi segmentleyiniz.

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_

clusters_kmeans = kmeans.labels_

df = pd.read_csv(r"C:\Users\Batuhan\Desktop\MIUUL\3. Hafta\FLOCLTVPrediction/flo_data_20k.csv")

df["cluster"] = clusters_kmeans

df["cluster"] = df["cluster"] + 1

# Ad�m 4: Herbir segmenti istatistiksel olarak inceleyeniz.

liste = df["cluster"].unique()
liste.sort()

df["TotalPrice"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

for i in liste:
    print("K�me s�n�f�:", i)
    print("-"*120)
    print((df[df["cluster"] == i].describe().T).round(2))
    print("-"*120, "\n\n")

df.groupby("cluster")["TotalPrice"].mean().plot(kind="line")
plt.xlabel("Number of Cluster")
plt.ylabel("Total Price Mean")
plt.title("Average Sales Price by Clusters")
plt.show()



# G�REV 3: Hierarchical Clustering ile M��teri Segmentasyonu

# Ad�m 1: G�rev 2'de stand�rla�t�rd���n�z dataframe'i kullanarak optimum k�me say�s�n� belirleyiniz.

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

hc_average = linkage(df, "average")

plt.figure(figsize=(40, 20))
plt.title("Hiyerar�ik K�meleme Dendogram�")
plt.xlabel("G�zlem Birimleri")
plt.ylabel("Uzakl�klar")
dendrogram(hc_average, leaf_font_size=10)
plt.show()

# Ad�m 2: Modelinizi olu�turunuz ve m��terileriniz segmentleyiniz.

# k�me say�s�n�n belirlenmesi
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


# Ad�m 3: Her bir segmenti istatistiksel olarak inceleyeniz.

for i in liste:
    print("K�me s�n�f�:", i)
    print("-"*120)
    print((df[df["hi_cluster_no"] == i].describe().T).round(2))
    print("-"*120, "\n\n")