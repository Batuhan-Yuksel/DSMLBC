# MAKÝNE ÖÐRENMESÝ ÝLE YETENEK AVCILIÐI SINIFLANDIRMA

# ÝÞ PROBLEMÝ

# Scout’lar tarafýndan izlenen futbolcularýn özelliklerine verilen puanlara göre, oyuncularýn hangi sýnýf
# (average, highlighted) oyuncu olduðunu tahminleme.

# Veri Seti Hikayesi

# Veri seti Scoutium’dan maçlarda gözlemlenen futbolcularýn özelliklerine göre scoutlarýn deðerlendirdikleri futbolcularýn, maç
# içerisinde puanlanan özellikleri ve puanlarýný içeren bilgilerden oluþmaktadýr.


# scoutium_attributes.csv
# task_response_id Bir scoutun bir maçta bir takýmýn kadrosundaki tüm oyunculara dair deðerlendirmelerinin kümesi
# match_id Ýlgili maçýn id'si
# evaluator_id Deðerlendiricinin(scout'un) id'si
# player_id Ýlgili oyuncunun id'si
# position_id Ýlgili oyuncunun o maçta oynadýðý pozisyonun id’si
# 1: Kaleci
# 2: Stoper
# 3: Sað bek
# 4: Sol bek
# 5: Defansif orta saha
# 6: Merkez orta saha
# 7: Sað kanat
# 8: Sol kanat
# 9: Ofansif orta saha
# 10: Forvet
# analysis_id Bir scoutun bir maçta bir oyuncuya dair özellik deðerlendirmelerini içeren küme
# attribute_id Oyuncularýn deðerlendirildiði her bir özelliðin id'si
# attribute_value Bir scoutun bir oyuncunun bir özelliðine verdiði deðer(puan)

# scoutium_potential_labels.csv
# task_response_id Bir scoutun bir maçta bir takýmýn kadrosundaki tüm oyunculara dair deðerlendirmelerinin kümesi
# match_id Ýlgili maçýn id'si
# evaluator_id Deðerlendiricinin(scout'un) id'si
# player_id Ýlgili oyuncunun id'si
# potential_label Bir scoutun bir maçta bir oyuncuyla ilgili nihai kararýný belirten etiket. (hedef deðiþken)

from warnings import filterwarnings
import warnings
filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# Görevler

# Adým 1: scoutium_attributes.csv ve scoutium_potential_labels.csv dosyalarýný okutunuz.
attributes = pd.read_csv(r"C:\Users\Batuhan\Desktop\scoutium_attributes.csv", sep=";")
potential = pd.read_csv(r"C:\Users\Batuhan\Desktop\scoutium_potential_labels.csv", sep=";")



# Adým 2: Okutmuþ olduðumuz csv dosyalarýný merge fonksiyonunu kullanarak birleþtiriniz.
# ("task_response_id", 'match_id', 'evaluator_id' "player_id" 4 adet deðiþken üzerinden birleþtirme iþlemini gerçekleþtiriniz.)

df_ = pd.merge(attributes, potential, on=["task_response_id", 'match_id', 'evaluator_id', "player_id"])
df = df_.copy()

# Adým 3: position_id içerisindeki Kaleci (1) sýnýfýný veri setinden kaldýrýnýz.

df = df[~(df["position_id"] == 1)]
df.reset_index(inplace=True, drop=True)

# Adým 4: potential_label içerisindeki below_average sýnýfýný veri setinden kaldýrýnýz.( below_average sýnýfý tüm verisetinin %1'ini oluþturur)

df = df[~(df["potential_label"] == "below_average")]
df.reset_index(inplace=True, drop=True)

# Adým 5: Oluþturduðunuz veri setinden “pivot_table” fonksiyonunu kullanarak bir tablo oluþturunuz. Bu pivot table'da her satýrda bir oyuncu
# olacak þekilde manipülasyon yapýnýz
    # Adým 1: Ýndekste “player_id”,“position_id” ve “potential_label”, sütunlarda “attribute_id” ve deðerlerde scout’larýn oyunculara verdiði
    # puan “attribute_value” olacak þekilde pivot table’ý oluþturunuz.

pivot_df = df.pivot_table(index=["player_id","position_id","potential_label"], columns="attribute_id", values="attribute_value")

    # Adým 2: “reset_index” fonksiyonunu kullanarak indeksleri deðiþken olarak atayýnýz ve “attribute_id” sütunlarýnýn isimlerini
    # stringe çeviriniz.

pivot_df.reset_index(inplace=True)

for i in range(0, len(pivot_df.columns)):
    pivot_df.rename(columns={pivot_df.columns[i]: str(pivot_df.columns[i])}, inplace=True)

df = pivot_df[:]

# Adým 6: Label Encoder fonksiyonunu kullanarak “potential_label” kategorilerini (average, highlighted) sayýsal olarak ifade ediniz.

from sklearn.preprocessing import LabelEncoder, StandardScaler

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

df = label_encoder(df, "potential_label")

# Adým 7: Sayýsal deðiþken kolonlarýný “num_cols” adýyla bir listeye atayýnýz.


for i in range(0, len(df["position_id"])):
    df["position_id"][i] = str(df["position_id"][i])

df["position_id"] = df["position_id"].astype(str)

def grab_col_names(dataframe, cat_th=8, car_th=20):
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
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols = [col for col in cat_cols if "potential_label" not in col]
cat_cols.append("position_id")

df["position_id"] = df["position_id"].astype(str)

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols)



num_cols = [col for col in num_cols  if "player_id" not in col]
num_cols = [col for col in num_cols if "position_id" not in col]

# Adým 8: Kaydettiðiniz bütün “num_cols” deðiþkenlerindeki veriyi ölçeklendirmek için StandardScaler uygulayýnýz.


ss = StandardScaler()
df[num_cols] = ss.fit_transform(df[num_cols])



# Adým 9: Elimizdeki veri seti üzerinden minimum hata ile futbolcularýn potansiyel etiketlerini tahmin eden bir makine öðrenmesi modeli
# geliþtiriniz. (Roc_auc, f1, precision, recall, accuracy metriklerini yazdýrýnýz.)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


y = df["potential_label"]
X = df.drop(["player_id", "potential_label"], axis=1)

def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=5, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

base_models(X, y, scoring="accuracy")

# 0.8856 (RF)
# 0.8672 (CatBoost)
# 0.8671 (LightGBM)
# 0.8599 (LR)
# 0.8562 (XGBoost)



rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, 9],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.7, 1]}

lightgbm_params = {"learning_rate": [0.01, 0.02, 0.05, 0.1],
               "n_estimators": [200, 300, 350, 400],
               "colsample_bytree": [0.9, 0.8, 1]}


catboost_params = {"iterations": [200,500],
                    "learning_rate": [0.01,0.05, 0.1],
                    "depth": [3,5,8] }

log_params = {"penalty":["l1", "l2", "elasticnet"],
              "solver": ["lbfgs", "liblinear", "newton-cg"]}


classifiers = [('CatBoost', CatBoostClassifier(verbose=False), catboost_params),
               ('LR', LogisticRegression(), log_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params)]





def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

best_models = hyperparameter_optimization(X, y, 5)

def voting_classifier(best_models, X, y):
    print("Voting Classifier...")

    voting_clf = VotingClassifier(estimators=[('XGBoost', best_models["XGBoost"]),
                                              ('RF', best_models["RF"]),
                                              ('CatBoost', best_models["CatBoost"])],
                                  voting='soft').fit(X, y)
    cv_results = cross_validate(voting_clf, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    print(f"Precision: {cv_results['test_precision'].mean()}")
    print(f"Recall: {cv_results['test_recall'].mean()}")
    return voting_clf

voting_clf = voting_classifier(best_models, X, y)

from sklearn.metrics import classification_report, roc_auc_score






