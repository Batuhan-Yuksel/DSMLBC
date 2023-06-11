# # # # # # # # # # # # # # # # # #
# # # # # BATUHAN YÜKSEL # # # # #
# # # # # # # # # # # # # # # # # #

import numpy as np
import pandas as pd
import seaborn as sns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
telco = pd.read_csv(r"C:\Users\Batuhan\Desktop\Telco-Customer-Churn.csv")
# telco = pd.read_csv("C:\\Users\Batuhan\\Desktop\\Telco-Customer-Churn.csv")
df = telco.copy()

df.shape
df.dtypes
df.head()
df.tail()
df.isnull().values.any()
df.isnull().sum()
df.isnull().sum().sum()
df.describe().T
for col in df.columns:
    if df[col].dtype == ("float64") or df[col].dtype == ("int64"):
        print(col)
df["SeniorCitizen"].unique()

num_gender = [0 if col == "Male" else 1 for col in df["gender"]]

df["NEW_PaperlessBilling"] = df["PaperlessBilling"].apply(lambda x:"Evt" if x == "Yes" else "Hyr")
df["PaperlessBilling"].head()
df["NEW_PaperlessBilling"].head()

def check_online(col):
    if col == "Yes":
        return "Evet"
    elif col == "No":
        return "Hayır"
    else:
        return "İnterneti_yok"

online = df.filter(like="Online")
for col in online:
    df[col] = df[col].apply(lambda x: check_online(x))

df[df["TotalCharges"] < 30]
df["TotalCharges"].dtype
# df["TotalCharges"] = df["TotalCharges"].astype('float64')
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce") # çalışan bu
# df["TotalCharges"] = pd.to_numeric(df["TotalCharges"]) # ikinci yol aynı sonuç
df["TotalCharges"].dtype
df[df["TotalCharges"] < 30]
df.head()

df[df["PaymentMethod"] == "Electronic check"]["MonthlyCharges"].mean()

df.groupby("PaymentMethod")["MonthlyCharges"].mean()

df[(df["gender"] == "Female") & ((df["InternetService"] == "Fiber optic") | (df["InternetService"] == "DSL"))]["MonthlyCharges"].sum()

df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)
df["Churn"].head()

df.groupby(["Contract","PhoneService"])["Churn"].mean()

df.pivot_table(index="Contract", columns="PhoneService", values="Churn", aggfunc="mean")

df["tenure"].head()
df["tenure"].dtype
df["tenure"].unique()
df["tenure"].nunique()
df["tenure"].min()
df["tenure"].max()

df["NEWC_tenure"] = pd.cut(df["tenure"],bins=[0,12,22,32,42,52,62,72],labels=["Cat_1","Cat_2","Cat_3","Cat_4","Cat_5","Cat_6","Cat_7"])


