##################################
# # # # # BATUHAN YÜKSEL # # # # #
##################################

# Association Rule Based Recommender System

# İş Problemi
# Türkiye’nin en büyük online hizmet platformu olan Armut, hizmet verenler ile hizmet almak isteyenleri buluşturmaktadır.
# Bilgisayarın veya akıllı telefonunun üzerinden birkaç dokunuşla temizlik, tadilat, nakliyat gibi hizmetlere kolayca ulaşılmasını sağlamaktadır.
# Hizmet alan kullanıcıları ve bu kullanıcıların almış oldukları servis ve kategorileri içeren veri setini kullanarak
# Association Rule Learning ile ürün tavsiye sistemi oluşturulmak istenmektedir.

# Veri Seti Hikayesi
# Veri seti müşterilerin aldıkları servislerden ve bu servislerin kategorilerinden oluşmaktadır. Alınan her hizmetin tarih ve saat
# bilgisini içermektedir.
# UserId: Müşteri numarası
# ServiceId:
# Her kategoriye ait anonimleştirilmiş servislerdir. (Örnek : Temizlik kategorisi altında koltuk yıkama servisi)
# Bir ServiceId farklı kategoriler altında bulanabilir ve farklı kategoriler altında farklı servisleri ifade eder.
# (Örnek: CategoryId’si 7 ServiceId’si 4 olan hizmet petek temizliği iken CategoryId’si 2 ServiceId’si 4 olan hizmet mobilya montaj)
# CategoryId: Anonimleştirilmiş kategorilerdir. (Örnek : Temizlik, nakliyat, tadilat kategorisi)
# CreateDate: Hizmetin satın alındığı tarih


import numpy as np
import pandas as pd
import datetime
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

df_ = pd.read_csv(r"C:\Users\Batuhan\Desktop\armut_data.csv")
df = df_.copy()

df.head()
df.info()
df["Hizmet"] = df["ServiceId"].astype("str") + "_" + df["CategoryId"].astype("str")


import datetime

df["CreateDate"] = pd.to_datetime(df["CreateDate"], format="%Y-%m-%d")
#df["CreateDate"] = df["CreateDate"].dt.strftime("%Y-%m")
df["SepetId"] = df["UserId"].astype("str") + "_" + df["CreateDate"].dt.strftime("%Y-%m")

# GÖREV 2: Birliktelik Kuralları Üretiniz ve Öneride bulununuz

# Adım 1: Aşağıdaki gibi sepet, hizmet pivot table’i oluşturunuz.

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

frequent_itemsets = apriori(pivot_df,
                            min_support=0.01,
                            use_colnames=True)

frequent_itemsets.sort_values("support", ascending=False) # sıraladık

rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)

rules[(rules["support"]>0.01) & (rules["confidence"]>0.01) & (rules["lift"]>1)]. \
sort_values("confidence", ascending=False)

def arl_recommender(rules_df, hizmet, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == hizmet:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]


arl_recommender(rules, "2_0", 1)
