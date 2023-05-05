##################################
# # # # # BATUHAN Y�KSEL # # # # #
##################################

# Association Rule Based Recommender System

# �� Problemi
# T�rkiye�nin en b�y�k online hizmet platformu olan Armut, hizmet verenler ile hizmet almak isteyenleri bulu�turmaktad�r.
# Bilgisayar�n veya ak�ll� telefonunun �zerinden birka� dokunu�la temizlik, tadilat, nakliyat gibi hizmetlere kolayca ula��lmas�n� sa�lamaktad�r.
# Hizmet alan kullan�c�lar� ve bu kullan�c�lar�n alm�� olduklar� servis ve kategorileri i�eren veri setini kullanarak
# Association Rule Learning ile �r�n tavsiye sistemi olu�turulmak istenmektedir.

# Veri Seti Hikayesi
# Veri seti m��terilerin ald�klar� servislerden ve bu servislerin kategorilerinden olu�maktad�r. Al�nan her hizmetin tarih ve saat
# bilgisini i�ermektedir.
# UserId: M��teri numaras�
# ServiceId:
# Her kategoriye ait anonimle�tirilmi� servislerdir. (�rnek : Temizlik kategorisi alt�nda koltuk y�kama servisi)
# Bir ServiceId farkl� kategoriler alt�nda bulanabilir ve farkl� kategoriler alt�nda farkl� servisleri ifade eder.
# (�rnek: CategoryId�si 7 ServiceId�si 4 olan hizmet petek temizli�i iken CategoryId�si 2 ServiceId�si 4 olan hizmet mobilya montaj)
# CategoryId: Anonimle�tirilmi� kategorilerdir. (�rnek : Temizlik, nakliyat, tadilat kategorisi)
# CreateDate: Hizmetin sat�n al�nd��� tarih

# Proje G�revleri
import numpy as np
import pandas as pd
import datetime
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

# G�REV 1: Veriyi Haz�rlama

# Ad�m 1: armut_data.csv dosyas�n� okutunuz.

df_ = pd.read_csv(r"C:\Users\Batuhan\Desktop\armut_data.csv")
df = df_.copy()
# Ad�m 2: ServisID her bir CategoryID �zelinde farkl� bir hizmeti temsil etmektedir. ServiceID ve CategoryID�yi "_" ile birle�tirerek ,
# bu hizmetleri temsil edecek yeni bir de�i�ken olu�turunuz.

df.head()
df.info()
df["Hizmet"] = df["ServiceId"].astype("str") + "_" + df["CategoryId"].astype("str")

# Ad�m 3: Veri seti hizmetlerin al�nd��� tarih ve saatten olu�maktad�r, herhangi bir sepet tan�m� (fatura vb. ) bulunmamaktad�r. Association Rule
# Learning uygulayabilmek i�in bir sepet (fatura vb.) tan�m� olu�turulmas� gerekmektedir. Burada sepet tan�m� her bir m��terinin ayl�k ald���
# hizmetlerdir. �rne�in; 7256 id'li m��teri 2017'in 8.ay�nda ald��� 9_4, 46_4 hizmetleri bir sepeti; 2017�in 10.ay�nda ald��� 9_4, 38_4 hizmetleri
# ba�ka bir sepeti ifade etmektedir. Sepetleri unique bir ID ile tan�mlanmas� gerekmektedir. Bunun i�in �ncelikle sadece y�l ve ay i�eren yeni bir
# date de�i�keni olu�turunuz. UserID ve yeni olu�turdu�unuz date de�i�kenini "_" ile birle�tirirek ID ad�nda yeni bir de�i�kene atay�n�z.

import datetime

df["CreateDate"] = pd.to_datetime(df["CreateDate"], format="%Y-%m-%d")
#df["CreateDate"] = df["CreateDate"].dt.strftime("%Y-%m")
df["SepetId"] = df["UserId"].astype("str") + "_" + df["CreateDate"].dt.strftime("%Y-%m")

# G�REV 2: Birliktelik Kurallar� �retiniz ve �neride bulununuz

# Ad�m 1: A�a��daki gibi sepet, hizmet pivot table�i olu�turunuz.

pivot_df = df.pivot_table(index="SepetId", columns="Hizmet").fillna(0). \
    droplevel(0, axis=1).applymap(lambda x: 1 if x > 1 else 0)

pivot_df = df.groupby(['SepetId', 'Hizmet']). \
    agg({"Hizmet": "count"}). \
    unstack(). \
    fillna(0). \
    applymap(lambda x: 1 if x > 0 else 0)

# Hizmet 0_8 10_9 11_11 12_7 13_11 14_7..
# SepetID
# 0_2017-08 0 0 0 0 0 0..
# 0_2017-09 0 0 0 0 0 0..
# 0_2018-01 0 0 0 0 0 0..
# 0_2018-04 0 0 0 0 0 1..
# 10000_2017-08 0 0 0 0 0 0..
df[df["UserId"] == 0]

# Ad�m 2: Birliktelik kurallar�n� olu�turunuz.

frequent_itemsets = apriori(pivot_df,
                            min_support=0.01,
                            use_colnames=True)
# support de�eri min 0.01 olan t�m birliktelikleri bulduk.
# bu bulunanlar her bir itemin olasl���d�r.
frequent_itemsets.sort_values("support", ascending=False) # s�ralad�k

rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)

rules[(rules["support"]>0.01) & (rules["confidence"]>0.01) & (rules["lift"]>1)]. \
sort_values("confidence", ascending=False)

# Ad�m 3: arl_recommender fonksiyonunu kullanarak en son 2_0 hizmetini alan bir kullan�c�ya hizmet �nerisinde bulununuz.


def arl_recommender(rules_df, hizmet, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == hizmet:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]


arl_recommender(rules, "2_0", 1)
