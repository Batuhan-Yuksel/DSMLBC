##################################
# # # # # BATUHAN YÜKSEL # # # # #
##################################


# ÝÞ PROBLEMÝ

# Özellikleri belirtildiðinde kiþilerin diyabet hastasý olup olmadýklarýný tahmin  edebilecek bir makine öðrenmesi modeli geliþtirilmesi
# istenmektedir. Modeli  geliþtirmeden önce gerekli olan veri analizi ve özellik mühendisliði adýmlarýný gerçekleþtirmeniz beklenmektedir

# Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalýklarý Enstitüleri'nde tutulan büyük veri setinin parçasýdýr. ABD'deki
# Arizona Eyaleti'nin en büyük 5. þehri olan Phoenix þehrinde yaþayan 21 yaþ ve üzerinde olan Pima Indian kadýnlarý üzerinde
# yapýlan diyabet araþtýrmasý için kullanýlan verilerdir.
# Hedef deðiþken "outcome" olarak belirtilmiþ olup; 1 diyabet test sonucunun pozitif oluþunu, 0 ise negatif oluþunu belirtmektedir.

# Pregnancies: Hamilelik sayýsý
# Glucose: Oral glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu
# Blood Pressure: Kan Basýncý (Küçük tansiyon) (mm Hg)
# SkinThickness: Cilt Kalýnlýðý
# Insulin: 2 saatlik serum insülini (mu U/ml)
# DiabetesPedigreeFunction: Fonksiyon (Oral glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu)
# BMI: Vücut kitle endeksi
# Age: Yaþ (yýl)
# Outcome: Hastalýða sahip (1) ya da deðil (0)

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# PROJE GÖREVLERÝ

# GÖREV 1 : Keþifçi Veri Analizi
# Adým 1: Genel resmi inceleyiniz.

df = pd.read_csv(r"C:\Users\Batuhan\Desktop\MIUUL\7. Hafta\diabetes\diabetes.csv")
df.shape
df.head()
df.info()
df.dtypes
df.isnull().values.any()
df.isnull().sum()
df.isnull().sum().sum()
df.describe().T

# Adým 2: Numerik ve kategorik deðiþkenleri yakalayýnýz.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal deðiþkenlerin isimlerini verir.
    Not: Kategorik deðiþkenlerin içerisine numerik görünümlü kategorik deðiþkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Deðiþken isimleri alýnmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan deðiþkenler için sýnýf eþik deðeri
        car_th: int, optinal
                kategorik fakat kardinal deðiþkenler için sýnýf eþik deðeri

    Returns
    ------
        cat_cols: list
                Kategorik deðiþken listesi
        num_cols: list
                Numerik deðiþken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal deðiþken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam deðiþken sayýsý
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamý toplam deðiþken sayýsýna eþittir: cat_cols + num_cols + cat_but_car = deðiþken sayýsý

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    # kategorik deðiþkenlerde gez ama kategorik görünüp aslýnda kardinal olanlarý alma diyoruz.
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    # veri tipi object'ten farklý olanlarý getirdik.
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    # sayýsal deðiþkenlerde gez ama sayýsal görünüp aslýnda kategorik olanlarý çýkar diyoruz.
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in df.columns:
    print("Deðiþken:", col, "----->   Eþsiz deðerlerinin sayýsý:", df[col].nunique())

for col in df.columns:
    print("Deðiþken:", col, "----->   Eþsiz deðerlerinin sayýsý:", len(df[col].unique()))

# Kategorik deðiþken ve hedef deðiþken:  Outcome
# Numerik deðiþkenler: 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'

# Adým 3: Numerik ve kategorik deðiþkenlerin analizini yapýnýz.

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

# Adým 4: Hedef deðiþken analizi yapýnýz. (Kategorik deðiþkenlere göre hedef deðiþkenin ortalamasý, hedef deðiþkene göre
# numerik deðiþkenlerin ortalamasý)

# Tek kategorik deðiþken ayný zamanda hedef deðiþkendir.
# hedef deðiþkene göre numerik deðiþkenlerin ortalamasý
df.groupby("Outcome")[num_cols].mean()

# Adým 5: Aykýrý gözlem analizi yapýnýz.

df.describe().T
df.describe([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T

# eþik deðerleri belirleyelim
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

outlier_thresholds(df, num_cols)

# aykýrý deðer kontrolü
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

check_outlier(df, num_cols)

for col in num_cols:
    print(col, check_outlier(df, col))
# Sonuç: True. Aykýrý deðer var.

# Adým 6: Eksik gözlem analizi yapýnýz.

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    # ilk olarak eksik gözlem barýndýran deðiþkenler seçildi
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    # daha sonra sayýlarýný ve yüzdelik deðerlerini bir df içine atalým
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name: # na_name argümaný eksik deðerlerin barýndýðý deðiþkenlerin isimlerini ver dedik eðer True girilirse verir
        return na_columns

missing_values_table(df)

# Eksik gözlem yoktur.

# Adým 7: Korelasyon analizi yapýnýz.

corr_df = df.corr()
corr_df.sort_values(by="Outcome",ascending=False)
# Outcome ve glucose deðiþkeni arasýnda %46.7'lik bir iliþki var.

# GÖREV 2: Feature Engineering

# Adým 1: Eksik ve aykýrý deðerler için gerekli iþlemleri yapýnýz. Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb.
# deðiþkenlerde 0 deðeri içeren gözlem birimleri eksik deðeri ifade ediyor olabilir. Örneðin; bir kiþinin glikoz veya insulin deðeri 0
# olamayacaktýr. Bu durumu dikkate alarak sýfýr deðerlerini ilgili deðerlerde NaN olarak atama yapýp sonrasýnda eksik
# deðerlere iþlemleri uygulayabilirsiniz.


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def change_integer(y):
    x = y + 1

x = 5
change_integer(8)

# aykýrý deðer kontrolü
for col in num_cols:
    print(col, check_outlier(df, col))

# aykýrý deðerleri baskýlama
for col in num_cols:
    replace_with_thresholds(df, col)

# aykýrý deðerleri tekrar kontrol edelim
for col in num_cols:
    print(col, check_outlier(df, col))
# aykýrý deðerler baskýlandý

# pregnancies deðiþkeninde 0 deðeri olabilir o yüzden num_cols içinden çýkartýlýr.
num_cols = [col for col in num_cols if col not in "Pregnancies"]

# 0 deðerlerini NaN ile deðiþtirelim
# for col in num_cols:
#    df[col].replace(0, np.nan, inplace=True)

def zero_to_nan(dataframe, num_cols):
    for col in num_cols:
        dataframe[col].replace(0, np.nan, inplace=True)

zero_to_nan(df,num_cols)
# 0 deðerleri NaN ile deðiþtirildi
df.isnull().sum()

# SkinThickness ve Insulin deðiþkeninde eksik deðerler var
for col in num_cols:
    if df[col].isnull().sum() < (df.shape[0]/10):
        df = df.loc[~(df[col].isnull())]
    else:
        df[col] = df[col].fillna(df.groupby("Outcome")[col].transform("mean"))
        # df[col] = df[col].fillna(df.groupby("Pregnancies")[col].transform("mean"))
df.groupby("Outcome")["Insulin", "SkinThickness"].mean()
df.isnull().sum()
# eksik deðerler giderildi.

# Adým 2: Yeni deðiþkenler oluþturunuz.

df.head()

df["Age_Cat"] = pd.cut(df["Age"], bins=[df["Age"].min(),36,55,df["Age"].max()+1], labels=["Young","Adult","Old"], right=False)

df["NumberOfPregnancy"] = pd.cut(df["Pregnancies"], bins=[df["Pregnancies"].min(),4,8,df["Pregnancies"].max()+1],right=False,
                                 labels=["Normal","Çok","Aþýrý"])

def calculate_bmi(col):
    if col["BMI"] < 18.5:
        return "Under"
    elif col["BMI"] >= 18.5 and col["BMI"] <= 24.9:
        return "Healthy"
    elif col["BMI"] >= 25 and col["BMI"] <= 29.9:
        return "Over"
    elif col["BMI"] >= 30:
        return "Obese"

df = df.assign(Result_of_BMI=df.apply(calculate_bmi, axis=1))

df.loc[(df['BMI'] < 18.5) & (df['Age'] < 56), 'NEW_AGE_BMI'] = 'matureunderweight'
df.loc[(df['BMI'] >= 18.5) & (df['Age'] < 56), 'NEW_AGE_BMI'] = 'maturehealthyweight'
df.loc[(df['BMI'] >= 25) & (df['Age'] < 56), 'NEW_AGE_BMI'] = 'matureoverweight'
df.loc[(df['BMI'] >= 30) & (df['Age'] < 56), 'NEW_AGE_BMI'] = 'matureobese'

df.loc[(df['BMI'] < 18.5) & (df['Age'] >= 56), 'NEW_AGE_BMI'] = 'seniorunderweight'
df.loc[(df['BMI'] >= 18.5) & (df['Age'] >= 56), 'NEW_AGE_BMI'] = 'seniorhealthyweight'
df.loc[(df['BMI'] >= 25) & (df['Age'] >= 56), 'NEW_AGE_BMI'] = 'senioroverweight'
df.loc[(df['BMI'] >= 30) & (df['Age'] >= 56), 'NEW_AGE_BMI'] = 'seniorobese'

df.head()

# Adým 3: Encoding iþlemlerini gerçekleþtiriniz.

df["Age_Cat"].value_counts()
df["NumberOfPregnancy"].value_counts()


def count_of_values(dataframe):
    for col in dataframe:
        print(dataframe[col].value_counts())
new_variables = df[["NumberOfPregnancy", "Result_of_BMI", "Age_Cat", 'NEW_AGE_BMI']]
count_of_values(new_variables)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
cat_summary(df, "Age_Cat")
cat_summary(df, "NumberOfPregnancy")
# Rare Encoding yapmaya gerek yok

# One Hot Encoding
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, ohe_cols)
df.head()
cat_cols, num_cols, cat_but_car = grab_col_names(df)


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "Outcome", cat_cols)

# iki sýnýflý olup sýnýflarýn bir tanesinin oraný 0.01'den az olan var mý ?
useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]
# Yok

# Adým 4: Numerik deðiþkenler için standartlaþtýrma yapýnýz.
check_outlier(df, num_cols)
df.describe().T
for col in num_cols:
    print(col, check_outlier(df, col))
# Standart scaler

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

# MinMaxScaler()
# mms = MinMaxScaler()
# for col in num_cols:
#    df[col + "_min_max_scaler"] = mms.fit_transform(df[[col]])

# Adým 5: Model oluþturunuz.
df.head()
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=22)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)
# %86 doðruluk

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X_train)