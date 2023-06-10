#####################################################
# Demand Forecasting
#####################################################

# 3-month sales forecast

import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


########################
# Loading the data
########################

train = pd.read_csv(r'C:\Users\Batuhan\Desktop\time_series\datasets\demand_forecasting/train.csv', parse_dates=['date'])
test = pd.read_csv(r'C:\Users\Batuhan\Desktop\time_series\datasets\demand_forecasting/test.csv', parse_dates=['date'])

sample_sub = pd.read_csv(r'C:\Users\Batuhan\Desktop\time_series\datasets\demand_forecasting/sample_submission.csv')

df = pd.concat([train, test], sort=False)

#####################################################
# FEATURE ENGINEERING
#####################################################

def create_date_features(df):
    df['month'] = df.date.dt.month # ay bilgisi
    df['day_of_month'] = df.date.dt.day # ayýn günü
    df['day_of_year'] = df.date.dt.dayofyear # yýlýn hangi günü
    df['week_of_year'] = df.date.dt.weekofyear # yýlýn hangi haftasý
    df['day_of_week'] = df.date.dt.dayofweek # haftanýn hangi günü
    df['year'] = df.date.dt.year # yýl bilgisi
    df["is_wknd"] = df.date.dt.weekday // 4 # hafta sonu mu ?
    df['is_month_start'] = df.date.dt.is_month_start.astype(int) # ayýn baþlangýcý mý ?
    df['is_month_end'] = df.date.dt.is_month_end.astype(int) # ayýn sonu mu ?
    return df

df = create_date_features(df)

df.groupby(["store", "item", "month"]).agg({"sales": ["sum", "mean", "median", "std"]})



########################
# Random Noise
########################

def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))


########################
# Lag/Shifted Features
########################

df.sort_values(by=['store', 'item', 'date'], axis=0, inplace=True)


def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])

########################
# Rolling Mean Features
########################

def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe

df = roll_mean_features(df, [365, 546])

########################
# Exponentially Weighted Mean Features
########################

def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]

df = ewm_features(df, alphas, lags)


########################
# One-Hot Encoding
########################

df = pd.get_dummies(df, columns=['store', 'item', 'day_of_week', 'month'])

########################
# Converting sales to log(1+sales)
########################

df['sales'] = np.log1p(df["sales"].values) # 1 ifadesi olasý bazý hatalarýn önüne geçmek için bir metottur.

#####################################################
# Model
#####################################################

########################
# Custom Cost Function
########################

# SMAPE: Symmetric mean absolute percentage error (adjusted MAPE)

def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val


def lgbm_smape(preds, train_data):
    labels = train_data.get_label() # label'lar gerçek deðerler, preds tahmin deðerleri
    smape_val = smape(np.expm1(preds), np.expm1(labels)) # ikisini karþýlaþtýrýyoruz ve smape skourunu hesaplayacaðýz
    return 'SMAPE', smape_val, False

########################
# Time-Based Validation Sets
########################
# train data
train = df.loc[(df["date"] < "2017-01-01"), :]

# validation data
val = df.loc[(df["date"] >= "2017-01-01") & (df["date"] < "2017-04-01"), :]

cols = [col for col in train.columns if col not in ['date', 'id', "sales", "year"]]

Y_train = train['sales']
X_train = train[cols]

Y_val = val['sales']
X_val = val[cols]

########################
# Time Series Model with LightGBM
########################

lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1000,
              'early_stopping_rounds': 200,
              'nthread': -1}

lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols) # train seti bu þekilde oluþturulur bu veri yapýsýnda

lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols) # validasyon seti bu þekilde oluþturulur bu veri yapýsýnda

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape,
                  verbose_eval=100)

y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(Y_val))


########################
# Feature Importance
########################

def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))
    return feat_imp

plot_lgb_importances(model, num=200)

plot_lgb_importances(model, num=30, plot=True)

feat_imp = plot_lgb_importances(model, num=200)

importance_zero = feat_imp[feat_imp["gain"] == 0]["feature"].values

imp_feats = [col for col in cols if col not in importance_zero]



########################
# Final Model
########################
# With all the data
train = df.loc[~df.sales.isna()]
Y_train = train['sales']
X_train = train[cols]


test = df.loc[df.sales.isna()]
X_test = test[cols]

lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}

lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

final_model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)

test_preds = final_model.predict(X_test, num_iteration=model.best_iteration)

# Final Predicted Sales
predicted_sales = pd.DataFrame(np.expm1(test_preds))
predicted_sales.rename(columns={0:"Prediceted Sales"})



