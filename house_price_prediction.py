##################################
# # # # # BATUHAN Y�KSEL # # # # #
##################################

# EV F�YAT TAHM�N MODEL�

# Her bir eve ait �zelliklerin ve ev fiyatlar�n�n bulundu�u veri seti kullan�larak, farkl� tipteki evlerin fiyatlar�na ili�kin bir
# makine ��renmesi projesi ger�ekle�tirilmek istenmektedir.

# Veri Seti Hikayesi

# Ames, Lowa�daki konut evlerinden olu�an bu veri seti i�erisinde 79 a��klay�c� de�i�ken bulunduruyor. Kaggle �zerinde bir yar��mas� da
# bulunan projenin veri seti ve yar��ma sayfas�na a�a��daki linkten ula�abilirsiniz.
# Veri seti bir kaggle yar��mas�na ait oldu�undan dolay� train ve test olmak �zere iki farkl� csv dosyas� vard�r.
# Test veri setinde ev fiyatlar� bo� b�rak�lm�� olup, bu de�erleri sizin tahmin etmeniz beklenmektedir.


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# G�rev 1: Ke�if�i Veri Analizi
# Ad�m 1: Train ve Test veri setlerini okutup birle�tiriniz. Birle�tirdi�iniz veri �zerinden ilerleyiniz.

train = pd.read_csv(r"C:\Users\Batuhan\Desktop\machine_learning\datasets\train.csv")
test = pd.read_csv(r"C:\Users\Batuhan\Desktop\machine_learning\datasets\test.csv")

df_ = pd.concat([train, test])
df = df_.copy()
df.head()
df.tail()

# Ad�m 2: Numerik ve kategorik de�i�kenleri yakalay�n�z.

def grab_col_names(dataframe, cat_th=11, car_th=30):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal de�i�kenlerin isimlerini verir.
    Not: Kategorik de�i�kenlerin i�erisine numerik g�r�n�ml� kategorik de�i�kenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                De�i�ken isimleri al�nmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan de�i�kenler i�in s�n�f e�ik de�eri
        car_th: int, optinal
                kategorik fakat kardinal de�i�kenler i�in s�n�f e�ik de�eri

    Returns
    ------
        cat_cols: list
                Kategorik de�i�ken listesi
        num_cols: list
                Numerik de�i�ken listesi
        cat_but_car: list
                Kategorik g�r�n�ml� kardinal de�i�ken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam de�i�ken say�s�
        num_but_cat cat_cols'un i�erisinde.
        Return olan 3 liste toplam� toplam de�i�ken say�s�na e�ittir: cat_cols + num_cols + cat_but_car = de�i�ken say�s�

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    # kategorik de�i�kenlerde gez ama kategorik g�r�n�p asl�nda kardinal olanlar� alma diyoruz.
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    # veri tipi object'ten farkl� olanlar� getirdik.
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    # say�sal de�i�kenlerde gez ama say�sal g�r�n�p asl�nda kategorik olanlar� ��kar diyoruz.
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col != "Id"]

num_cols = [col for col in num_cols if col != "SalePrice"]

# Ad�m 3: Gerekli d�zenlemeleri yap�n�z. (Tip hatas� olan de�i�kenler gibi)

for col in num_cols:
    print(col, str(df[col].dtype))
# num_cols i�inde tip hatas� olan yok

for col in cat_cols:
    print(col, str(df[col].dtype), df[col].nunique())

df[cat_cols] = df[cat_cols].astype(object)

# cat_cols i�indeki say�sal de�i�kenler kategorik de�i�kene �evrildi.

# Ad�m 4: Numerik ve kategorik de�i�kenlerin veri i�indeki da��l�m�n� g�zlemleyiniz.

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

# Ad�m 5: Kategorik de�i�kenler ile hedef de�i�ken incelemesini yap�n�z.



# Ad�m 6: Ayk�r� g�zlem var m� inceleyiniz.

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
    print(col, check_outlier(df, col))

# Ayk�r� g�zlem var

# Ad�m 7: Eksik g�zlem var m� inceleyiniz.

df.isnull().values.any()
df.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df, True)
# Eksik g�zlem var

# G�REV 2: Feature Engineering

# Ad�m 1: Eksik ve ayk�r� g�zlemler i�in gerekli i�lemleri yap�n�z.
train.isnull().sum()

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_thresholds(df, col)
# num_cols i�indeki ayk�r� de�erler giderildi

# sat�� fiyat� ile en y�ksek korelasyona sahip olan OverallQual de�i�keni k�r�l�m�nda num_cols de�i�kenleri ortalama ile doldurulmal�

for col in num_cols:
    if df[col].dtype == "int64":
        df[col] = df[col].fillna(df.groupby("OverallQual")[col].transform("mean").round())
    else:
        df[col] = df[col].fillna(df.groupby("OverallQual")[col].transform("mean"))

for col in num_cols:
    print(col, df[col].isnull().sum())
# num_cols i�indeki eksik de�erler giderildi

for col in cat_cols:
    print(col, df[col].isnull().sum())

liste = []

for col in cat_cols:
    if df[col].isnull().sum() / df.shape[0] > 0.8:
        liste.append(col)
        df = df.drop(col, axis=1)
    else:
       df[col] = df[col].fillna(df[col].mode()[0])

cat_cols = list(set(cat_cols).difference(set(liste)))

for col in cat_cols:
    print(col, df[col].isnull().sum())


# Ad�m 2: Rare Encoder uygulay�n�z.

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "SalePrice", cat_cols)

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

new_df = rare_encoder(df, 0.1)
df = new_df

df = df.drop("PoolArea", axis=1)
num_cols.remove("PoolArea")

# Ad�m 3: Yeni de�i�kenler olu�turunuz.

dff = pd.concat([df[num_cols], df["SalePrice"]], axis=1)

dff.corr().sort_values(by="SalePrice", ascending=False)

df["GrLivArea"].describe().T

# yeni de�i�ken 1
df["Yap�m-Tadilat"] = df["YearRemodAdd"] - df["YearBuilt"]
df["Yap�m-Tadilat"] = df["Yap�m-Tadilat"].astype(float)

# yeni de�i�ken 2
df["GrLivArea_Cat"] = pd.cut(df["GrLivArea"], bins=[df["GrLivArea"].min(), df["GrLivArea"].quantile(0.25),
                                                    df["GrLivArea"].quantile(0.5), df["GrLivArea"].quantile(0.75),
                                                    df["GrLivArea"].max()+1], right=False,
                             labels=["Category_1", "Category_2", "Category_3", "Category_4"])
df["GrLivArea_Cat"].dtype
# yeni de�i�ken 3
df["YearBuilt_Cat"] = pd.cut(df["YearBuilt"], bins=[df["YearBuilt"].min(), df["YearBuilt"].quantile(0.25),
                                                    df["YearBuilt"].median(), df["YearBuilt"].quantile(0.75),
                                                    df["YearBuilt"].max()+1], right=False,
                             labels=["Category_1", "Category_2", "Category_3", "Category_4"])

# yeni de�i�ken 4
df["Multiple_Overall"] = df["OverallQual"] * df["OverallCond"]
df["Multiple_Overall"] = df["Multiple_Overall"].astype(float)

# Ad�m 4: Encoding i�lemlerini ger�ekle�tiriniz.

# Label Encoding
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype not in ["int64", "float64"] and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

# One-Hot Encoding

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)
df.head()

# Standartla�t�rma

ss = StandardScaler()

for col in num_cols:
    df[col] = ss.fit_transform(df[[col]])


# G�REV 3: Model Kurma

# Ad�m 1:  Train ve Test verisini ay�r�n�z. (SalePrice de�i�keni bo� olan de�erler test verisidir.)

df.reset_index(inplace=True, drop=True)
variables = [col for col in df.columns if col not in "SalePrice"]
len(variables)

# df.loc[:len(train), variables]
# df.loc[:1459, "SalePrice"] # y_train
# df.loc[:1459, variables] # x_train
# df.loc[1460:, "SalePrice"] # y_test
# df.loc[1460:, variables] # x_test


X_train = df.loc[:1459, variables]
X_test = df.loc[1460:, variables]
y_train = df.loc[:1459, "SalePrice"]
y_test = df.loc[1460:, "SalePrice"]

from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.metrics import mean_squared_error

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                    test_size=0.25,
                                                    random_state=1)


# Ad�m 2:  Train verisi ile model kurup, model ba�ar�s�n� de�erlendiriniz.

# XGBoost

from xgboost import XGBRegressor

xgb = XGBRegressor()
xgb_model = xgb.fit(X_train, y_train)
y_pred = xgb_model.predict(X_val)
np.sqrt(mean_squared_error(y_val, y_pred))
# 32796.44469421341



# LightGBM

from lightgbm import LGBMRegressor

lgbm = LGBMRegressor()

lgbm_model = lgbm.fit(X_train, y_train)

y_pred = lgbm_model.predict(X_val)
np.sqrt(mean_squared_error(y_val, y_pred))
# 26947.13417036209

# 33663.596754615544

# CatBoost

from catboost import CatBoostRegressor

catb = CatBoostRegressor()
catb_model = catb.fit(X_train, y_train)

y_pred = catb_model.predict(X_val)
np.sqrt(mean_squared_error(y_val, y_pred))
# 25025.90238329884




# Ad�m 3: Hiperparemetre optimizasyonu ger�ekle�tiriniz.

# XGBoost

xgb_grid = {
     'colsample_bytree': [0.4, 0.5,0.6,0.9,1],
     'n_estimators':[100, 200, 500, 1000],
     'max_depth': [2,3,4,5,6],
     'learning_rate': [0.1, 0.01, 0.5]}

xgb = XGBRegressor()

xgb_cv = GridSearchCV(xgb,
                      param_grid = xgb_grid,
                      cv = 10,
                      n_jobs = -1,
                      verbose = 1)

xgb_cv.fit(X_train, y_train)

xgb_cv.best_params_

xgb = XGBRegressor(colsample_bytree = 0.6,
                         learning_rate = 0.1,
                         max_depth = 3,
                         n_estimators = 1000)

xgb_final = xgb.fit(X_train,y_train)
y_pred = xgb_final.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# 30920.890687451956

# LightGBM

lgbm_grid = {
    'colsample_bytree': [0.4, 0.5,0.6,0.9,1],
    'learning_rate': [0.01, 0.1, 0.5,1],
    'n_estimators': [20, 40, 100, 200, 500,1000],
    'max_depth': [1,2,3,4,5,6,7,8]}

lgbm = LGBMRegressor()

lgbm_cv_model = GridSearchCV(lgbm, lgbm_grid, cv=10, n_jobs = -1, verbose = 1)

lgbm_cv_model.fit(X_train, y_train)

lgbm_cv_model.best_params_

lgbm = LGBMRegressor(learning_rate = 0.01,
                           max_depth = 5,
                           n_estimators = 1000,
                          colsample_bytree = 0.4)

lgbm_final = lgbm.fit(X_train,y_train)
y_pred = lgbm_final.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# 32430.02722135586

# Catboost

catb_grid = {
    'iterations': [200,500,1000,2000],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'depth': [3,4,5,6,7,8]}

catb = CatBoostRegressor()

catb_cv_model = GridSearchCV(catb, catb_grid, cv=10, n_jobs = -1, verbose = 1)
catb_cv_model.fit(X_train, y_train)
catb_cv_model.best_params_
catb_tuned = CatBoostRegressor(iterations = 200,
                               learning_rate = 0.01,
                               depth = 8)

catb_tuned = catb_tuned.fit(X_train,y_train)
y_pred = catb_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))