##########################################
# # # # # # # BATUHAN Y�KSEL # # # # # # #
##########################################

# Regresyon Modelleri i�in Hata De�erlendirme

import pandas as pd

df = pd.DataFrame([[5, 7, 3, 3, 2, 7, 3, 10, 6, 4, 8, 1, 1, 9, 1],
                   [600, 900, 550, 500, 400, 950, 540, 1200, 900, 550, 1100, 460, 400, 1000, 380]]).T


df.columns = ["Deneyim Y�l� (x)", "Maa� (y)"]

# G�REV

# �al��anlar�n deneyim y�l� ve maa� bilgileri verilmi�tir.

# 1 - Verilen bias ve weight�e g�re do�rusal regresyon model denklemini olu�turunuz. Bias = 275, Weight = 90 (y�=b+wx)

# 2 - Olu�turdu�unuz model denklemine g�re tablodaki t�m deneyim y�llar� i�in maa� tahmini yap�n�z.

# 3 - Modelin ba�ar�s�n� �l�mek i�in MSE, RMSE, MAE skorlar�n� hesaplay�n�z.

# 1 -
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


X = df[["Deneyim Y�l� (x)"]]
y = df[["Maa� (y)"]]

reg_model = LinearRegression().fit(X, y)

reg_model.intercept_[0]

reg_model.coef_[0][0]

# Model denklemi:

# y_hat = reg_model.intercept_[0] + reg_model.coef_[0][0] * x
# y_hat = 274.35602094240846 + 90.20942408376962 * x
# y_hat = 275 + 90 * x


# 2 -
exp = list(df["Deneyim Y�l� (x)"])

tahmin = []
for x in exp:
    y_hat = 275 + 90 * x
    tahmin.append(y_hat)

tahmin = pd.DataFrame(tahmin, columns=["Maa� Tahmini (y')"])
df = pd.concat([df, tahmin], axis=1)

# MAE
mae = mean_absolute_error(df["Maa� (y)"], df["Maa� Tahmini (y')"])
# 54.33

# MSE
mse = mean_squared_error(df["Maa� (y)"], df["Maa� Tahmini (y')"])
# 4438.33

# RMSE
rmse = np.sqrt(mean_squared_error(df["Maa� (y)"], df["Maa� Tahmini (y')"]))
# 66.62


df["Hata (y-y')"] = df["Maa� (y)"] - df["Maa� Tahmini (y')"]

df["Hata Kareleri"] = df["Hata (y-y')"] ** 2

df["Mutlak Hata (|y-y'|)"] = abs(df["Hata (y-y')"])

