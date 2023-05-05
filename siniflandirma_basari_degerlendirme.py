##########################################
# # # # # # # BATUHAN Y�KSEL # # # # # # #
##########################################

# SINIFLANDIRMA MODEL� DE�ERLEND�RME
import pandas as pd

# G�REVLER

# G�REV 1: M��terinin churn olup olmama durumunu tahminleyen bir s�n�fland�rma modeli olu�turulmu�tur. 10 test verisi g�zleminin
# ger�ek de�erleri ve modelin tahmin etti�i olas�l�k de�erleri verilmi�tir.
# - E�ik de�erini 0.5 alarak confusion matrix olu�turunuz.
# - Accuracy, Recall, Precision, F1 Skorlar�n� hesaplay�n�z.

df = pd.DataFrame({"Ger�ek De�er": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                   "S�n�f = 1 Olas�l���": [0.7, 0.8, 0.65, 0.9, 0.45, 0.5, 0.55, 0.35, 0.4, 0.25]})

df["Tahmin"] = [1 if i > 0.5 else 0 for i in df["S�n�f = 1 Olas�l���"]]

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# confusion matrix
confusion_matrix(df["Ger�ek De�er"], df["Tahmin"], labels=[1,0])

# accuracy
accuracy_score(df["Ger�ek De�er"], df["Tahmin"])

print(classification_report(df["Ger�ek De�er"], df["Tahmin"]))

# precision : 0.8
# recall : 0.67
# f1 score : 0.73

#           | S�n�f = 1 | S�n�f = 0 |
# S�n�f = 1 |   a = 4       b = 2   |    6     |
# S�n�f = 0 |   c = 1       d = 3   |    4     |
#           |     5     |     5     |

accuracy = (4 + 3) / 10
recall = 4 / (4 + 2)
precision = 4 / (4 + 1)
f1_score = 2 * (precision * recall) / (precision + recall)

# G�REV 2
# Banka �zerinden yap�lan i�lemler s�ras�nda doland�r�c�l�k i�lemlerinin yakalanmas� amac�yla s�n�fland�rma modeli olu�turulmu�tur.
# %90.5 do�ruluk oran� elde edilen modelin ba�ar�s� yeterli bulunup model canl�ya al�nm��t�r.
# Ancak canl�ya al�nd�ktan sonra modelin ��kt�lar� beklendi�i gibi olmam��, i� birimi modelin ba�ar�s�z oldu�unu iletmi�tir.
# A�a��da modelin tahmin sonu�lar�n�n karma��kl�k matriksi verilmi�tir. Buna g�re;


#               | Fraud (1)  | Non-Fraud (0) |
# Fraud (1)     |     5      |      5        |    10
# Non-Fraud (0) |    90      |     900       |    990
#               |    95      |     905       |    1000


# 1. Accuracy, Recall, Precision, F1 Skorlar�n� hesaplay�n�z.

accuracy = (5 + 900) / 1000
precision = 5 / (5 + 90)
recall = 5 / (5 + 5)
f1_score = 2 * (precision * recall) / (precision + recall)


# 2. Veri Bilimi ekibinin g�zden ka��rd��� durum ne olabilir yorumlay�n�z.

# Veri seti dengesiz bir yap�ya sahip yani s�n�flar aras�nda dengesizlik sorunu var.
# Bunu gidermek i�in "Oversampling" yap�lmal�d�r. Oversampling y�ntemlerinden "SMOTE" kullan�labilir.