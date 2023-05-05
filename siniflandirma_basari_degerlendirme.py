##########################################
# # # # # # # BATUHAN YÜKSEL # # # # # # #
##########################################

# SINIFLANDIRMA MODELÝ DEÐERLENDÝRME
import pandas as pd

# GÖREVLER

# GÖREV 1: Müþterinin churn olup olmama durumunu tahminleyen bir sýnýflandýrma modeli oluþturulmuþtur. 10 test verisi gözleminin
# gerçek deðerleri ve modelin tahmin ettiði olasýlýk deðerleri verilmiþtir.
# - Eþik deðerini 0.5 alarak confusion matrix oluþturunuz.
# - Accuracy, Recall, Precision, F1 Skorlarýný hesaplayýnýz.

df = pd.DataFrame({"Gerçek Deðer": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                   "Sýnýf = 1 Olasýlýðý": [0.7, 0.8, 0.65, 0.9, 0.45, 0.5, 0.55, 0.35, 0.4, 0.25]})

df["Tahmin"] = [1 if i > 0.5 else 0 for i in df["Sýnýf = 1 Olasýlýðý"]]

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# confusion matrix
confusion_matrix(df["Gerçek Deðer"], df["Tahmin"], labels=[1,0])

# accuracy
accuracy_score(df["Gerçek Deðer"], df["Tahmin"])

print(classification_report(df["Gerçek Deðer"], df["Tahmin"]))

# precision : 0.8
# recall : 0.67
# f1 score : 0.73

#           | Sýnýf = 1 | Sýnýf = 0 |
# Sýnýf = 1 |   a = 4       b = 2   |    6     |
# Sýnýf = 0 |   c = 1       d = 3   |    4     |
#           |     5     |     5     |

accuracy = (4 + 3) / 10
recall = 4 / (4 + 2)
precision = 4 / (4 + 1)
f1_score = 2 * (precision * recall) / (precision + recall)

# GÖREV 2
# Banka üzerinden yapýlan iþlemler sýrasýnda dolandýrýcýlýk iþlemlerinin yakalanmasý amacýyla sýnýflandýrma modeli oluþturulmuþtur.
# %90.5 doðruluk oraný elde edilen modelin baþarýsý yeterli bulunup model canlýya alýnmýþtýr.
# Ancak canlýya alýndýktan sonra modelin çýktýlarý beklendiði gibi olmamýþ, iþ birimi modelin baþarýsýz olduðunu iletmiþtir.
# Aþaðýda modelin tahmin sonuçlarýnýn karmaþýklýk matriksi verilmiþtir. Buna göre;


#               | Fraud (1)  | Non-Fraud (0) |
# Fraud (1)     |     5      |      5        |    10
# Non-Fraud (0) |    90      |     900       |    990
#               |    95      |     905       |    1000


# 1. Accuracy, Recall, Precision, F1 Skorlarýný hesaplayýnýz.

accuracy = (5 + 900) / 1000
precision = 5 / (5 + 90)
recall = 5 / (5 + 5)
f1_score = 2 * (precision * recall) / (precision + recall)


# 2. Veri Bilimi ekibinin gözden kaçýrdýðý durum ne olabilir yorumlayýnýz.

# Veri seti dengesiz bir yapýya sahip yani sýnýflar arasýnda dengesizlik sorunu var.
# Bunu gidermek için "Oversampling" yapýlmalýdýr. Oversampling yöntemlerinden "SMOTE" kullanýlabilir.