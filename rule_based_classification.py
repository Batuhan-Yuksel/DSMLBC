import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.width", 1000)
pd.set_option('display.max_rows', None)
pd.set_option("display.max_columns", None)

df = pd.read_csv("persona.csv")
df.head()
df.info()
df.ndim
df.shape
df.dtypes
df.describe().T

df["SOURCE"].nunique()
df["SOURCE"].value_counts()

df["PRICE"].nunique()
df["PRICE"].unique()

df["PRICE"].value_counts()

df["COUNTRY"].value_counts()

df.groupby("COUNTRY")["PRICE"].sum()

df["SOURCE"].value_counts()

df.groupby("COUNTRY")["PRICE"].mean()

df.groupby("SOURCE")["PRICE"].mean()

df.pivot_table(index="SOURCE", columns="COUNTRY", values="PRICE", aggfunc="mean")

df.pivot_table(index=["COUNTRY", "SOURCE", "SEX", "AGE"], aggfunc="mean").head()

agg_df = df.pivot_table(index=["COUNTRY", "SOURCE", "SEX", "AGE"], aggfunc="mean")
agg_df.sort_values(by="PRICE", ascending=False, inplace=True)
agg_df.head()

agg_df.reset_index(inplace=True)
agg_df.head()
agg_df.columns

agg_df["AGE_CAT"] = pd.cut(x=agg_df["AGE"], bins=[0,19,25,35,45,70])
agg_df.head()



liste = []
for i in agg_df["AGE"]:
    if i < 20:
        liste.append("0_19")
    elif 20 <= i < 26:
        liste.append("20_25")
    elif 26 <= i < 36:
        liste.append("26_35")
    elif 36 <= i < 46:
        liste.append("36_45")
    else:
        liste.append("46_70")
liste[0:20]
agg_df["AGE"].head(10)
agg_df.head(20)

agg_df["AGE_CAT_2"] = liste

liste = list(zip(agg_df["COUNTRY"],agg_df["SOURCE"], agg_df["SEX"], agg_df["AGE_CAT_2"]))
agg_df["customers_level_based"] = [i[0].upper() + "_" + i[1].upper() + "_" + i[2].upper() + "_" + i[3] for i in liste]
agg_df.head(20)

new_df = agg_df[["customers_level_based", "PRICE"]]
new_df.head()

new_df2 = new_df.groupby("customers_level_based")["PRICE"].mean()
new_df2 = pd.DataFrame(new_df2)
new_df2.head()
new_df2.reset_index(inplace=True)
new_df2.columns

new_df2["SEGMENT"] = pd.cut(new_df2["PRICE"], 4, labels=["D", "C", "B", "A"])
new_df2.head()

new_df2.groupby("SEGMENT")["PRICE"].agg(["mean", "max", "sum"])

# Prediction
new_user = "TUR_ANDROID_FEMALE_26_35"

new_df2[new_df2["customers_level_based"] == new_user] # ???

new_user_2 = "FRA_IOS_FEMALE_26_35"

new_df2[new_df2["customers_level_based"] == new_user_2] # ???


