##################################
# # # # # BATUHAN YÜKSEL # # # # #
##################################

# TELCO CUSTOMER CHURN



# CustomerId: Müþteri Ýd’si
# Gender: Cinsiyet
# SeniorCitizen: Müþterinin yaþlý olup olmadýðý (1, 0)
# Partner: Müþterinin bir ortaðý olup olmadýðý (Evet, Hayýr)
# Dependents: Müþterinin bakmakla yükümlü olduðu kiþiler olup olmadýðý (Evet, Hayýr
# tenure: Müþterinin þirkette kaldýðý ay sayýsý
# PhoneService: Müþterinin telefon hizmeti olup olmadýðý (Evet, Hayýr)
# MultipleLines: Müþterinin birden fazla hattý olup olmadýðý (Evet, Hayýr, Telefon hizmeti yok)
# InternetService: Müþterinin internet servis saðlayýcýsý (DSL, Fiber optik, Hayýr)
# OnlineSecurity: Müþterinin çevrimiçi güvenliðinin olup olmadýðý (Evet, Hayýr, Ýnternet hizmeti yok)
# OnlineBackup: Müþterinin online yedeðinin olup olmadýðý (Evet, Hayýr, Ýnternet hizmeti yok)
# DeviceProtection: Müþterinin cihaz korumasýna sahip olup olmadýðý (Evet, Hayýr, Ýnternet hizmeti yok)
# TechSupport: Müþterinin teknik destek alýp almadýðý (Evet, Hayýr, Ýnternet hizmeti yok)
# StreamingTV: Müþterinin TV yayýný olup olmadýðý (Evet, Hayýr, Ýnternet hizmeti yok)
# StreamingMovies: Müþterinin film akýþý olup olmadýðý (Evet, Hayýr, Ýnternet hizmeti yok)
# Contract: Müþterinin sözleþme süresi (Aydan aya, Bir yýl, Ýki yýl)
# PaperlessBilling: Müþterinin kaðýtsýz faturasý olup olmadýðý (Evet, Hayýr)
# PaymentMethod: Müþterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartý (otomatik))
# MonthlyCharges: Müþteriden aylýk olarak tahsil edilen tutar
# TotalCharges: Müþteriden tahsil edilen toplam tutar
# Churn: Müþterinin kullanýp kullanmadýðý (Evet veya Hayýr)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


telco = pd.read_csv(r"C:\Users\Batuhan\Desktop\TelcoChurn\Telco-Customer-Churn.csv")
df = telco.copy()
df.head()
df.info()
df.dtypes

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])
df.describe().T

for col in df.columns:
    print(" ",col, "nunique: ",df[col].nunique(), " ", df[col].dtype)

for col in df.columns:
    print("{} deðiþkeninin eþsiz deðer sayýsý: {}".format(col, df[col].nunique()))


cat_cols = [col for col in df.columns if df[col].nunique() < 10]

num_cols = [col for col in df.columns if df[col].nunique() >= 10 and df[col].dtype not in ["O"]]

num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtype in ["int64", "float64"]]

cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

cat_cols = [col for col in cat_cols if col not in cat_but_car]

cat_cols = cat_cols + num_but_cat

df.replace(" ", np.nan, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])
df["SeniorCitizen"] = df["SeniorCitizen"].astype(object)
df["Churn"] = [1 if i == "Yes" else 0 for i in df["Churn"]]
df["Churn"] = df["Churn"].astype(int)


def check_df(dataframe, head=5):
    print("----- Shape -----")
    print(dataframe.shape)
    print("----- Types -----")
    print(dataframe.dtypes)
    print("----- Head -----")
    print(dataframe.head(head))
    print("----- Tail -----")
    print(dataframe.tail(head))
    print("----- NA -----")
    print(dataframe.isnull().sum())
    print("----- Quantifiles -----")
    print(dataframe.describe([0.01, 0.05, 0.50, 0.95, 0.99]).T)

check_df(df[num_cols])
check_df(df[cat_cols])


for col in cat_cols:
    if col != "Churn":
        print(df.groupby("Churn")[col].value_counts())
        print("-------------------------------")


df.describe().T
df.describe([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T


sns.boxplot(data=df.loc[:, num_cols])
sns.boxplot(df["tenure"])
sns.boxplot(df["MonthlyCharges"])
sns.boxplot(df["TotalCharges"])


def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(check_outlier(df, col))

df.isnull().values.any()
df.isnull().sum()
df[df.iloc[:,:] == " "].sum()

# Feature Engineering

sns.boxplot(data=df, x=df["TotalCharges"])

sns.kdeplot(df["TotalCharges"])

df["TotalCharges"] = df["TotalCharges"].fillna(df.groupby("Churn")["TotalCharges"].transform("mean"))

df.isnull().values.any()

sns.boxplot(data=df, x=df["TotalCharges"])

df.corr().sort_values(by="Churn", ascending=False)

df["MonthlyCharges"].describe().T

df["MonthlyCharges_Cat"] = pd.cut(df["MonthlyCharges"], bins=[df["MonthlyCharges"].min(),43,68,93,df["MonthlyCharges"].max()+1],
                                  labels=["Lower", "Lower_Middle","Upper_Middle", "Upper"], right=False)

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

# Binary Encoding

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype not in ["int64", "float64"]
               and df[col].nunique() == 2]

for col in binary_cols:
    if col != "Churn":
        df = label_encoder(df, col)
df.head()

# One Hot Encoding

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)

df.head()

# Veri normal daðýlýma uyuyorsa standartscaler uymuyorsa minmax

# H0: Uygun
# H1: Deðil

from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal

for i in num_cols:
    test_stat, pvalue = shapiro(df.loc[:, i])
    print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df.head()
# Modelling

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


y = df["Churn"]

X = df.drop(["Churn","customerID"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=1)

# 1 - LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression

log = LogisticRegression()
log_model = log.fit(X_train,y_train)

y_pred = log_model.predict(X_test)
accuracy_score(y_test, y_pred)
# 0.8140525195173882

# 2 - KNN

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn_model = knn.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)
accuracy_score(y_test, y_pred)
# 0.7672107877927609

# 3 - CART

from sklearn.tree import DecisionTreeClassifier

cart = DecisionTreeClassifier()
cart_model = cart.fit(X_train, y_train)

y_pred = cart_model.predict(X_test)
accuracy_score(y_test, y_pred)
# 0.7423704755145494

# 4 - RANDOM FORESTS

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf_model = rf.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_test, y_pred)
# 0.794889992902768


# 5 - GBM

from sklearn.ensemble import GradientBoostingClassifier

gbm = GradientBoostingClassifier()
gbm_model = gbm.fit(X_train, y_train)
y_pred = gbm_model.predict(X_test)
accuracy_score(y_test, y_pred)
# 0.8055358410220014


# 6 - XGBoost

from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb_model = xgb.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
accuracy_score(y_test, y_pred)
# 0.8069552874378992


# 7 - LightGBM

from lightgbm import LGBMClassifier

lgbm = LGBMClassifier()
lgbm_model = lgbm.fit(X_train, y_train)
y_pred = lgbm_model.predict(X_test)
accuracy_score(y_test, y_pred)
# 0.808374733853797


# 8 - CatBoost

from catboost import CatBoostClassifier

cat = CatBoostClassifier()
cat_model = cat.fit(X_train, y_train)
y_pred = cat_model.predict(X_test)
accuracy_score(y_test, y_pred)
# 0.8105039034776437

modeller = [log_model, knn_model, cart_model, rf_model, gbm_model, xgb_model, lgbm_model, cat_model]

for model in modeller:
    print(model)
    y_pred = model.predict(X_test)
    dogruluk = accuracy_score(y_test, y_pred)
    print(dogruluk,"\n")

for model in modeller:
    isimler = model.__class__.__name__
    y_pred = model.predict(X_test)
    dogruluk = accuracy_score(y_test, y_pred)
    print("-"*28)
    print(isimler + ":" )
    print("Accuracy: {:.4%}".format(dogruluk))

# Seçilen modeller: GBM, XGBoost, LightGBM, CatBoost

# GBM Final

gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8, 10],
              "n_estimators": [100, 500, 1000],
              "subsample": [0.5, 0.7, 1]}

gbm = GradientBoostingClassifier()

gbm_best_grid = GridSearchCV(gbm, gbm_params, cv = 10, n_jobs = -1, verbose = 1)
gbm_best_grid.fit(X_train, y_train)

print("En iyi parametreler: " + str(gbm_best_grid.best_params_))
# en iyi parametrelerle final modeli
gbm = GradientBoostingClassifier(learning_rate = 0.01,
                                 max_depth = 3,
                                subsample= 0.5,
                                n_estimators = 1000)
gbm_final =  gbm.fit(X_train,y_train)
y_pred = gbm_final.predict(X_test)
accuracy_score(y_test, y_pred)
# 0.8076650106458482

# XGBoost Final

xgb_params = {"learning_rate": [0.01, 0.1],
                  "max_depth": [3, 5, 8],
                  "n_estimators": [100, 500, 1000, 2000],
                  "colsample_bytree": [0.7, 1]}

xgb = XGBClassifier()

xgb_best_grid = GridSearchCV(xgb, xgb_params, cv=10, n_jobs=-1, verbose=True)

xgb_best_grid.fit(X_train, y_train)

print("En iyi parametreler: " + str(xgb_best_grid.best_params_))

xgb = XGBClassifier(learning_rate = 0.01,
                    max_depth = 3,
                    n_estimators = 1000,
                    colsample_bytree = 1)

xgb_final =  xgb.fit(X_train,y_train)
y_pred = xgb_final.predict(X_test)
accuracy_score(y_test, y_pred)
# 0.8097941802696949

# LightGBM Final

lgbm_params = {"learning_rate": [0.01, 0.02, 0.05, 0.1],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 0.9, 1]}

lgbm = LGBMClassifier()

lgbm_best_grid = GridSearchCV(lgbm, lgbm_params, cv=10, n_jobs=-1, verbose=True)

lgbm_best_grid.fit(X_train, y_train)

print("En iyi parametreler: " + str(lgbm_best_grid.best_params_))

lgbm = LGBMClassifier(learning_rate = 0.01,
                        colsample_bytree=0.5,
                       n_estimators = 300)

lgbm_final = lgbm.fit(X_train,y_train)
y_pred = lgbm_final.predict(X_test)
accuracy_score(y_test, y_pred)
# 0.8147622427253371

# CatBoost Final

catb_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.05, 0.1],
                   "depth": [3, 5, 8]}

catb = CatBoostClassifier()

catb_best_grid = GridSearchCV(catb, catb_params, cv=10, n_jobs = -1, verbose = 1)

catb_best_grid.fit(X_train, y_train)

print("En iyi parametreler: " + str(catb_best_grid.best_params_))

catb =CatBoostClassifier(depth=5, iterations = 200, learning_rate = 0.05)

catb_final = catb.fit(X_train, y_train)

y_pred = catb_final.predict(X_test)
accuracy_score(y_test, y_pred)
# 0.8161816891412349
