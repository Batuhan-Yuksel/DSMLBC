#####################################################
# AB Testi ile Bidding Yöntemlerinin Dönüşümünün Karşılaştırılması
#####################################################


# Impression Reklam görüntüleme sayısı
# Click Görüntülenen reklama tıklama sayısı
# Purchase Tıklanan reklamlar sonrası satın alınan ürün sayısı
# Earning Satın alınan ürünler sonrası elde edilen kazanç

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

cont = pd.read_excel(r"C:\Users\Batuhan\Desktop/ab_testing.xlsx", sheet_name="Control Group")
test = pd.read_excel(r"C:\Users\Batuhan\Desktop/ab_testing.xlsx", sheet_name="Test Group")

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

df = pd.concat([cont, test], axis=0, ignore_index=True)

# HYPOTHESIS

# H0 : M1 = M2 --> Kontrol ve Test grupları arasında satın almaların ortalaması yönünden istatistiksel olarak anlamlı bir farklılık yoktur.
# H1 : M1!= M2 --> Fark vardır.

cont["Purchase"].mean()
test["Purchase"].mean()

# 1 - Normality:
# H0: Veriler normal dağılıma uymaktadır.
# H1: Veriler normal dağılıma uymamaktadır.

##########################################################  Density Plot   ##########################################################
purchase = pd.concat([cont["Purchase"], test["Purchase"]], axis=1)
purchase.columns = ["Purchase_Control", "Purchase_Test"]
sns.kdeplot(purchase)
plt.xlabel("Ortalama Satın Alma")
plt.ylabel("Yoğunluk")
plt.title("Normal Dağılıma Uygunluk")
##########################################################################################################################################


test_stat, pvalue = shapiro(cont["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

if pvalue < 0.05:
    print("{} < 0.05 olduğu için H0 Reddedilir veriler normal dağılıma uymamaktadır".format(pvalue))
else:
    print("{} > 0.05 olduğu için H0 Reddedilemez veriler normal dağılıma uymaktadır".format(pvalue))

test_stat, pvalue2 = shapiro(test["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue2))

if pvalue2 < 0.05:
    print("{} < 0.05 olduğu için H0 Reddedilir. Veriler normal dağılıma uymamaktadır".format(pvalue2))
else:
    print("{} > 0.05 olduğu için H0 Reddedilemez. Veriler normal dağılıma uymaktadır".format(pvalue2))

if (pvalue > 0.05) and (pvalue2 > 0.05):
    print("Normallik varsayımı sağlanmıştır\nVaryans homojenliği varsayımına geçiniz.")
else:
    print("Normallik varsayımı sağlanmamıştır\nNon-parametrik Mann Whitney-U testine geçiniz.")

# Variance Homogeneity
# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir

test_stat, pvalue = levene(cont["Purchase"], test["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

if pvalue < 0.05:
    print("{} < 0.05 olduğu için H0 Reddedilir Varyanslar homojen değildir".format(pvalue))
else:
    print("{} > 0.05 olduğu için H0 Reddedilemez varyanslar homojendir".format(pvalue))

# Varsayımlar sağlandığı için bağımsız iki örneklem testi yapılmalıdır.

test_stat, pvalue = ttest_ind(cont["Purchase"], test["Purchase"], equal_var=True)
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

if pvalue < 0.05:
    print(
        "{} < 0.05 olduğu için H0 Reddedilir İki grubun ortalamaları arasında istatistiksel olarak \n%95 güven düzeyinde "
        "anlamlı bir fark vardır.".format(pvalue))
else:
    print(
        "{} > 0.05 olduğu için H0 Reddedilemez İki grubun ortalamaları arasında istatistiksel olarak \n%95 güven düzeyinde "
        "anlamlı bir fark yoktur.".format(pvalue))

# %95 güven düzeyinde satın alma değişkeni açısından iki grubun ortalamaları arasında bir fark yoktur.
# Yani average bidding ve maximum bidding arasında satın alma ortalamaları açısından bir fark bulunamamıştır.




