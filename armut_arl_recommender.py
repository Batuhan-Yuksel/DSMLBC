##################################
# # # # # BATUHAN YÜKSEL # # # # #
##################################

# Association Rule Based Recommender System

# Ýþ Problemi
# Türkiye’nin en büyük online hizmet platformu olan Armut, hizmet verenler ile hizmet almak isteyenleri buluþturmaktadýr.
# Bilgisayarýn veya akýllý telefonunun üzerinden birkaç dokunuþla temizlik, tadilat, nakliyat gibi hizmetlere kolayca ulaþýlmasýný saðlamaktadýr.
# Hizmet alan kullanýcýlarý ve bu kullanýcýlarýn almýþ olduklarý servis ve kategorileri içeren veri setini kullanarak
# Association Rule Learning ile ürün tavsiye sistemi oluþturulmak istenmektedir.

# Veri Seti Hikayesi
# Veri seti müþterilerin aldýklarý servislerden ve bu servislerin kategorilerinden oluþmaktadýr. Alýnan her hizmetin tarih ve saat
# bilgisini içermektedir.
# UserId: Müþteri numarasý
# ServiceId:
# Her kategoriye ait anonimleþtirilmiþ servislerdir. (Örnek : Temizlik kategorisi altýnda koltuk yýkama servisi)
# Bir ServiceId farklý kategoriler altýnda bulanabilir ve farklý kategoriler altýnda farklý servisleri ifade eder.
# (Örnek: CategoryId’si 7 ServiceId’si 4 olan hizmet petek temizliði iken CategoryId’si 2 ServiceId’si 4 olan hizmet mobilya montaj)
# CategoryId: Anonimleþtirilmiþ kategorilerdir. (Örnek : Temizlik, nakliyat, tadilat kategorisi)
# CreateDate: Hizmetin satýn alýndýðý tarih

# Proje Görevleri
import numpy as np
import pandas as pd
import datetime
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

# GÖREV 1: Veriyi Hazýrlama

# Adým 1: armut_data.csv dosyasýný okutunuz.

df_ = pd.read_csv(r"C:\Users\Batuhan\Desktop\armut_data.csv")
df = df_.copy()
# Adým 2: ServisID her bir CategoryID özelinde farklý bir hizmeti temsil etmektedir. ServiceID ve CategoryID’yi "_" ile birleþtirerek ,
# bu hizmetleri temsil edecek yeni bir deðiþken oluþturunuz.

df.head()
df.info()
df["Hizmet"] = df["ServiceId"].astype("str") + "_" + df["CategoryId"].astype("str")

# Adým 3: Veri seti hizmetlerin alýndýðý tarih ve saatten oluþmaktadýr, herhangi bir sepet tanýmý (fatura vb. ) bulunmamaktadýr. Association Rule
# Learning uygulayabilmek için bir sepet (fatura vb.) tanýmý oluþturulmasý gerekmektedir. Burada sepet tanýmý her bir müþterinin aylýk aldýðý
# hizmetlerdir. Örneðin; 7256 id'li müþteri 2017'in 8.ayýnda aldýðý 9_4, 46_4 hizmetleri bir sepeti; 2017’in 10.ayýnda aldýðý 9_4, 38_4 hizmetleri
# baþka bir sepeti ifade etmektedir. Sepetleri unique bir ID ile tanýmlanmasý gerekmektedir. Bunun için öncelikle sadece yýl ve ay içeren yeni bir
# date deðiþkeni oluþturunuz. UserID ve yeni oluþturduðunuz date deðiþkenini "_" ile birleþtirirek ID adýnda yeni bir deðiþkene atayýnýz.

import datetime

df["CreateDate"] = pd.to_datetime(df["CreateDate"], format="%Y-%m-%d")
#df["CreateDate"] = df["CreateDate"].dt.strftime("%Y-%m")
df["SepetId"] = df["UserId"].astype("str") + "_" + df["CreateDate"].dt.strftime("%Y-%m")

# GÖREV 2: Birliktelik Kurallarý Üretiniz ve Öneride bulununuz

# Adým 1: Aþaðýdaki gibi sepet, hizmet pivot table’i oluþturunuz.

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

# Adým 2: Birliktelik kurallarýný oluþturunuz.

frequent_itemsets = apriori(pivot_df,
                            min_support=0.01,
                            use_colnames=True)
# support deðeri min 0.01 olan tüm birliktelikleri bulduk.
# bu bulunanlar her bir itemin olaslýðýdýr.
frequent_itemsets.sort_values("support", ascending=False) # sýraladýk

rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)

rules[(rules["support"]>0.01) & (rules["confidence"]>0.01) & (rules["lift"]>1)]. \
sort_values("confidence", ascending=False)

# Adým 3: arl_recommender fonksiyonunu kullanarak en son 2_0 hizmetini alan bir kullanýcýya hizmet önerisinde bulununuz.


def arl_recommender(rules_df, hizmet, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == hizmet:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]


arl_recommender(rules, "2_0", 1)
