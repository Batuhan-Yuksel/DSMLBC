#####################################################
# AB Testi ile Bidding Y�ntemlerinin D�n���m�n�n Kar��la�t�r�lmas�
#####################################################

# �� Problemi
# Facebook k�sa s�re �nce mevcut "maximum bidding" ad� verilen teklif verme t�r�ne alternatif olarak yeni bir teklif t�r� olan
# "average bidding"�i tan�tt�. M��terilerimizden biri olan bombabomba.com, bu yeni �zelli�i test etmeye karar verdi ve average bidding'in
# maximum bidding'den daha fazla d�n���m getirip getirmedi�ini anlamak i�in bir A/B testi yapmak istiyor. A/B testi 1 ayd�r devam ediyor
# ve bombabomba.com �imdi sizden bu A/B testinin sonu�lar�n� analiz etmenizi bekliyor. Bombabomba.com i�in nihai ba�ar� �l��t� Purchase'd�r. Bu
# nedenle, istatistiksel testler i�in Purchase metri�ine odaklan�lmal�d�r.

# Veri Seti Hikayesi

# Bir firman�n web site bilgilerini i�eren bu veri setinde kullan�c�lar�n g�rd�kleri ve t�klad�klar� reklam say�lar� gibi bilgilerin yan� s�ra
# buradan gelen kazan� bilgileri yer almaktad�r. Kontrol ve Test grubu olmak �zere iki ayr� veri seti vard�r. Bu veri setleri
# ab_testing.xlsx excel�inin ayr� sayfalar�nda yer almaktad�r. Kontrol grubuna Maximum Bidding, test grubuna Average Bidding uygulanm��t�r.

# Impression Reklam g�r�nt�leme say�s�
# Click G�r�nt�lenen reklama t�klama say�s�
# Purchase T�klanan reklamlar sonras� sat�n al�nan �r�n say�s�
# Earning Sat�n al�nan �r�nler sonras� elde edilen kazan�

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

# G�REV 1: Veriyi Haz�rlama ve Analiz Etme

# Ad�m 1: ab_testing_data.xlsx adl� kontrol ve test grubu verilerinden olu�an veri setini okutunuz. Kontrol ve test grubu verilerini ayr�
# de�i�kenlere atay�n�z.

cont = pd.read_excel(r"C:\Users\Batuhan\Desktop/ab_testing.xlsx", sheet_name="Control Group")
test = pd.read_excel(r"C:\Users\Batuhan\Desktop/ab_testing.xlsx", sheet_name="Test Group")

# Ad�m 2: Kontrol ve test grubu verilerini analiz ediniz.
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

# Ad�m 3: Analiz i�leminden sonra concat metodunu kullanarak kontrol ve test grubu verilerini birle�tiriniz.

df = pd.concat([cont, test], axis=0, ignore_index=True)

# G�REV  2: A/B Testinin Hipotezinin Tan�mlanmas�

# Ad�m 1: Hipotezi tan�mlay�n�z.

# H0 : M1 = M2 --> Kontrol ve Test gruplar� aras�nda sat�n almalar�n ortalamas� y�n�nden istatistiksel olarak anlaml� bir farkl�l�k yoktur.
# H1 : M1!= M2 --> Fark vard�r.

# Ad�m 2: Kontrol ve test grubu i�in purchase (kazan�) ortalamalar�n� analiz ediniz.

cont["Purchase"].mean()
test["Purchase"].mean()

# G�REV 3: Hipotez Testinin Ger�ekle�tirilmesi

# Ad�m 1: Hipotez testi yap�lmadan �nce varsay�m kontrollerini yap�n�z.
# Bunlar Normallik Varsay�m� ve Varyans Homojenli�idir. Kontrol ve test grubunun normallik varsay�m�na uyup uymad���n� Purchase de�i�keni
# �zerinden ayr� ayr� test ediniz.

# 1 - Normallik Varsay�m�:
# H0: Veriler normal da��l�ma uymaktad�r.
# H1: Veriler normal da��l�ma uymamaktad�r.

##########################################################  Yo�unluk Grafi�i   ##########################################################
purchase = pd.concat([cont["Purchase"], test["Purchase"]], axis=1)
purchase.columns = ["Purchase_Control", "Purchase_Test"]
sns.kdeplot(purchase)
plt.xlabel("Ortalama Sat�n Alma")
plt.ylabel("Yo�unluk")
plt.title("Normal Da��l�ma Uygunluk")
##########################################################################################################################################


test_stat, pvalue = shapiro(cont["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

if pvalue < 0.05:
    print("{} < 0.05 oldu�u i�in H0 Reddedilir veriler normal da��l�ma uymamaktad�r".format(pvalue))
else:
    print("{} > 0.05 oldu�u i�in H0 Reddedilemez veriler normal da��l�ma uymaktad�r".format(pvalue))

test_stat, pvalue2 = shapiro(test["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue2))

if pvalue2 < 0.05:
    print("{} < 0.05 oldu�u i�in H0 Reddedilir. Veriler normal da��l�ma uymamaktad�r".format(pvalue2))
else:
    print("{} > 0.05 oldu�u i�in H0 Reddedilemez. Veriler normal da��l�ma uymaktad�r".format(pvalue2))

if (pvalue > 0.05) and (pvalue2 > 0.05):
    print("Normallik varsay�m� sa�lanm��t�r\nVaryans homojenli�i varsay�m�na ge�iniz.")
else:
    print("Normallik varsay�m� sa�lanmam��t�r\nNon-parametrik Mann Whitney-U testine ge�iniz.")

# Varyans Homojenli�i Testi
# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen De�ildir

test_stat, pvalue = levene(cont["Purchase"], test["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

if pvalue < 0.05:
    print("{} < 0.05 oldu�u i�in H0 Reddedilir Varyanslar homojen de�ildir".format(pvalue))
else:
    print("{} > 0.05 oldu�u i�in H0 Reddedilemez varyanslar homojendir".format(pvalue))

# Ad�m 2: Normallik Varsay�m� ve Varyans Homojenli�i sonu�lar�na g�re uygun testi se�iniz.

# Varsay�mlar sa�land��� i�in ba��ms�z iki �rneklem testi yap�lmal�d�r.

# Ad�m 3: Test sonucunda elde edilen p_value de�erini g�z �n�nde bulundurarak kontrol ve test grubu sat�n alma ortalamalar� aras�nda istatistiki
# olarak anlaml� bir fark olup olmad���n� yorumlay�n�z.

test_stat, pvalue = ttest_ind(cont["Purchase"], test["Purchase"], equal_var=True)
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

if pvalue < 0.05:
    print(
        "{} < 0.05 oldu�u i�in H0 Reddedilir �ki grubun ortalamalar� aras�nda istatistiksel olarak \n%95 g�ven d�zeyinde "
        "anlaml� bir fark vard�r.".format(pvalue))
else:
    print(
        "{} > 0.05 oldu�u i�in H0 Reddedilemez �ki grubun ortalamalar� aras�nda istatistiksel olarak \n%95 g�ven d�zeyinde "
        "anlaml� bir fark yoktur.".format(pvalue))

# %95 g�ven d�zeyinde sat�n alma de�i�keni a��s�ndan iki grubun ortalamalar� aras�nda bir fark yoktur.
# Yani average bidding ve maximum bidding aras�nda sat�n alma ortalamalar� a��s�ndan bir fark bulunamam��t�r.

# G�REV 4: Sonu�lar�n Analizi

# Ad�m 1: Hangi testi kulland�n�z, sebeplerini belirtiniz.

# Ba��ms�z �ki �rneklem T Testi kullan�ld�. Ba��ms�z iki grubu ortalamalar� y�n�nden kar��la�t�rmak istedi�imiz i�in bu test kullan�ld�.

# Ad�m 2: Elde etti�iniz test sonu�lar�na g�re m��teriye tavsiyede bulununuz.

# Yorum: �ki grubun ortalamalar� aras�nda anlaml� bir fark yoktur. Average bidding ve maximum bidding y�ntemleri aras�nda sat�n almalar�
# etkilemesi a��s�ndan bir fark yoktur. Yeni y�nteme para yat�rmak bir �ey kazand�rmaz.

df = pd.read_csv(r"C:\Users\Batuhan\Desktop/course_reviews.csv")
df.loc[df["Progress"], "Rating"].mean()
df.loc[df["Progress"] <= 10, "Rating"].mean() * 0.22 + \
df.loc[(df["Progress"] > 10) & (df["Progress"] <= 45), "Rating"].mean() * 0.24 + \
df.loc[(df["Progress"] > 45) & (df["Progress"] <= 75), "Rating"].mean() * 0.26 + \
df.loc[(df["Progress"] > 75), "Rating"].mean() * 0.28

