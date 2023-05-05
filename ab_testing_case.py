#####################################################
# AB Testi ile Bidding Yöntemlerinin Dönüþümünün Karþýlaþtýrýlmasý
#####################################################

# Ýþ Problemi
# Facebook kýsa süre önce mevcut "maximum bidding" adý verilen teklif verme türüne alternatif olarak yeni bir teklif türü olan
# "average bidding"’i tanýttý. Müþterilerimizden biri olan bombabomba.com, bu yeni özelliði test etmeye karar verdi ve average bidding'in
# maximum bidding'den daha fazla dönüþüm getirip getirmediðini anlamak için bir A/B testi yapmak istiyor. A/B testi 1 aydýr devam ediyor
# ve bombabomba.com þimdi sizden bu A/B testinin sonuçlarýný analiz etmenizi bekliyor. Bombabomba.com için nihai baþarý ölçütü Purchase'dýr. Bu
# nedenle, istatistiksel testler için Purchase metriðine odaklanýlmalýdýr.

# Veri Seti Hikayesi

# Bir firmanýn web site bilgilerini içeren bu veri setinde kullanýcýlarýn gördükleri ve týkladýklarý reklam sayýlarý gibi bilgilerin yaný sýra
# buradan gelen kazanç bilgileri yer almaktadýr. Kontrol ve Test grubu olmak üzere iki ayrý veri seti vardýr. Bu veri setleri
# ab_testing.xlsx excel’inin ayrý sayfalarýnda yer almaktadýr. Kontrol grubuna Maximum Bidding, test grubuna Average Bidding uygulanmýþtýr.

# Impression Reklam görüntüleme sayýsý
# Click Görüntülenen reklama týklama sayýsý
# Purchase Týklanan reklamlar sonrasý satýn alýnan ürün sayýsý
# Earning Satýn alýnan ürünler sonrasý elde edilen kazanç

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# GÖREV 1: Veriyi Hazýrlama ve Analiz Etme

# Adým 1: ab_testing_data.xlsx adlý kontrol ve test grubu verilerinden oluþan veri setini okutunuz. Kontrol ve test grubu verilerini ayrý
# deðiþkenlere atayýnýz.

cont = pd.read_excel(r"C:\Users\Batuhan\Desktop/ab_testing.xlsx", sheet_name="Control Group")
test = pd.read_excel(r"C:\Users\Batuhan\Desktop/ab_testing.xlsx", sheet_name="Test Group")

# Adým 2: Kontrol ve test grubu verilerini analiz ediniz.
# control grubu
cont.head()
cont.info()
cont.shape
cont["Purchase"].mean()
cont["Purchase"].median()

# test grubu
test.head()
test.info()
test.shape
test["Purchase"].mean()
test["Purchase"].median()

# Adým 3: Analiz iþleminden sonra concat metodunu kullanarak kontrol ve test grubu verilerini birleþtiriniz.

df = pd.concat([cont, test], axis=0, ignore_index=True)

# GÖREV  2: A/B Testinin Hipotezinin Tanýmlanmasý

# Adým 1: Hipotezi tanýmlayýnýz.

# H0 : M1 = M2 --> Kontrol ve Test gruplarý arasýnda satýn almalarýn ortalamasý yönünden istatistiksel olarak anlamlý bir farklýlýk yoktur.
# H1 : M1!= M2 --> Fark vardýr.

# Adým 2: Kontrol ve test grubu için purchase (kazanç) ortalamalarýný analiz ediniz.

cont["Purchase"].mean()
test["Purchase"].mean()

# GÖREV 3: Hipotez Testinin Gerçekleþtirilmesi

# Adým 1: Hipotez testi yapýlmadan önce varsayým kontrollerini yapýnýz.
# Bunlar Normallik Varsayýmý ve Varyans Homojenliðidir. Kontrol ve test grubunun normallik varsayýmýna uyup uymadýðýný Purchase deðiþkeni
# üzerinden ayrý ayrý test ediniz.

# 1 - Normallik Varsayýmý:
# H0: Veriler normal daðýlýma uymaktadýr.
# H1: Veriler normal daðýlýma uymamaktadýr.

##########################################################  Yoðunluk Grafiði   ##########################################################
purchase = pd.concat([cont["Purchase"], test["Purchase"]], axis=1)
purchase.columns = ["Purchase_Control", "Purchase_Test"]
sns.kdeplot(purchase)
plt.xlabel("Ortalama Satýn Alma")
plt.ylabel("Yoðunluk")
plt.title("Normal Daðýlýma Uygunluk")
##########################################################################################################################################


test_stat, pvalue = shapiro(cont["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

if pvalue < 0.05:
    print("{} < 0.05 olduðu için H0 Reddedilir veriler normal daðýlýma uymamaktadýr".format(pvalue))
else:
    print("{} > 0.05 olduðu için H0 Reddedilemez veriler normal daðýlýma uymaktadýr".format(pvalue))

test_stat, pvalue2 = shapiro(test["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue2))

if pvalue2 < 0.05:
    print("{} < 0.05 olduðu için H0 Reddedilir. Veriler normal daðýlýma uymamaktadýr".format(pvalue2))
else:
    print("{} > 0.05 olduðu için H0 Reddedilemez. Veriler normal daðýlýma uymaktadýr".format(pvalue2))

if (pvalue > 0.05) and (pvalue2 > 0.05):
    print("Normallik varsayýmý saðlanmýþtýr\nVaryans homojenliði varsayýmýna geçiniz.")
else:
    print("Normallik varsayýmý saðlanmamýþtýr\nNon-parametrik Mann Whitney-U testine geçiniz.")

# Varyans Homojenliði Testi
# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Deðildir

test_stat, pvalue = levene(cont["Purchase"], test["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

if pvalue < 0.05:
    print("{} < 0.05 olduðu için H0 Reddedilir Varyanslar homojen deðildir".format(pvalue))
else:
    print("{} > 0.05 olduðu için H0 Reddedilemez varyanslar homojendir".format(pvalue))

# Adým 2: Normallik Varsayýmý ve Varyans Homojenliði sonuçlarýna göre uygun testi seçiniz.

# Varsayýmlar saðlandýðý için baðýmsýz iki örneklem testi yapýlmalýdýr.

# Adým 3: Test sonucunda elde edilen p_value deðerini göz önünde bulundurarak kontrol ve test grubu satýn alma ortalamalarý arasýnda istatistiki
# olarak anlamlý bir fark olup olmadýðýný yorumlayýnýz.

test_stat, pvalue = ttest_ind(cont["Purchase"], test["Purchase"], equal_var=True)
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

if pvalue < 0.05:
    print(
        "{} < 0.05 olduðu için H0 Reddedilir Ýki grubun ortalamalarý arasýnda istatistiksel olarak \n%95 güven düzeyinde "
        "anlamlý bir fark vardýr.".format(pvalue))
else:
    print(
        "{} > 0.05 olduðu için H0 Reddedilemez Ýki grubun ortalamalarý arasýnda istatistiksel olarak \n%95 güven düzeyinde "
        "anlamlý bir fark yoktur.".format(pvalue))

# %95 güven düzeyinde satýn alma deðiþkeni açýsýndan iki grubun ortalamalarý arasýnda bir fark yoktur.
# Yani average bidding ve maximum bidding arasýnda satýn alma ortalamalarý açýsýndan bir fark bulunamamýþtýr.

# GÖREV 4: Sonuçlarýn Analizi

# Adým 1: Hangi testi kullandýnýz, sebeplerini belirtiniz.

# Baðýmsýz Ýki Örneklem T Testi kullanýldý. Baðýmsýz iki grubu ortalamalarý yönünden karþýlaþtýrmak istediðimiz için bu test kullanýldý.

# Adým 2: Elde ettiðiniz test sonuçlarýna göre müþteriye tavsiyede bulununuz.

# Yorum: Ýki grubun ortalamalarý arasýnda anlamlý bir fark yoktur. Average bidding ve maximum bidding yöntemleri arasýnda satýn almalarý
# etkilemesi açýsýndan bir fark yoktur. Yeni yönteme para yatýrmak bir þey kazandýrmaz.

df = pd.read_csv(r"C:\Users\Batuhan\Desktop/course_reviews.csv")
df.loc[df["Progress"], "Rating"].mean()
df.loc[df["Progress"] <= 10, "Rating"].mean() * 0.22 + \
df.loc[(df["Progress"] > 10) & (df["Progress"] <= 45), "Rating"].mean() * 0.24 + \
df.loc[(df["Progress"] > 45) & (df["Progress"] <= 75), "Rating"].mean() * 0.26 + \
df.loc[(df["Progress"] > 75), "Rating"].mean() * 0.28

