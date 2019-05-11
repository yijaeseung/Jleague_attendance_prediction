
# coding: utf-8

import os
import pandas as pd
import numpy as np
import re
from urllib.request import urlretrieve
import requests
from bs4 import BeautifulSoup as BS
import matplotlib.pyplot as plt
import seaborn as sns
import googlemaps
from meteocalc import Temp, dew_point, heat_index
import math
from itertools import product
import jaconv
import warnings
import gc
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# Model

df = pd.read_csv(path + 'corr_and_encoding.py')

trn_df = df.loc[df['match_Year'] < 2016].sort_values(by='id')
val_df = df.loc[(df['match_Year'] == 2016)&(df['division'] == 1)].sort_values(by='id')
test_df = df.loc[df['match_Year'] >= 2017].sort_values(by='id')

trn_df = trn_df.reset_index().drop(['index'], 1)
val_df = val_df.reset_index().drop(['index'], 1)
test_df = test_df.reset_index().drop(['index'], 1)

val_ids = val_df.index
test_ids = test_df.index

#trn_df.shape, val_df.shape, test_df.shape

features = [c for c in trn_df.columns if c not in ['id', 'attendance']]
trn_y = np.log(trn_df['attendance'].values)
val_y = np.log(val_df['attendance'].values)

param = {'objective':'regression', 
           'n_estimators': 5000,
          'boosting_type':'gbdt',
           'subsample':0.83,
          'colsample_bytree':0.83,
              'reg_lambda': 0.8,
          'learning_rate':0.015,
           'seed':105,
           'random_state':105
         }

num_boost_round = 10000

fea_to_use = []
for feature in features:
    if 'lag4' not in feature:
        fea_to_use.append(feature)



oof = np.zeros(len(val_df))
predictions = np.zeros(len(test_df))

div1_index = trn_df.loc[(trn_df['match_Year'] > 1998)&(trn_df['division'] == 1)].index
trn_data = lgb.Dataset(trn_df.iloc[div1_index][fea_to_use], label = trn_y[div1_index])
val_data = lgb.Dataset(val_df[fea_to_use], label = val_y)



clf = lgb.train(param, trn_data, num_boost_round, valid_sets = [trn_data, val_data],
               verbose_eval = 200, early_stopping_rounds = 300, feval=evalerror)
oof[val_ids] = clf.predict(val_df[fea_to_use], num_iteration = clf.best_iteration)

feature_importance_df = pd.DataFrame()
feature_importance_df['Feature'] = fea_to_use
feature_importance_df['importance'] = clf.feature_importance()

predictions += clf.predict(test_df[fea_to_use], num_iteration = clf.best_iteration)
cv_score = rmse(val_y, oof)

print("CV Score: {:<8.5f}".format(cv_score))



best_features = feature_importance_df.loc[feature_importance_df.importance][:100]
# plt.figure(figsize=(14,50))
# sns.barplot(x='importance', y = 'Feature', data=best_features.sort_values(by='importance', ascending=False))
# plt.title("Feature importance (averaged/folds)")
# plt.tight_layout()
print(best_features)

