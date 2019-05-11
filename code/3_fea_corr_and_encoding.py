
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



# Feature Correlation

df = pd.read_csv(path + 'fea_eng2.csv')
#df.shape

corr_matrix = df.drop(['id'], 1).corr().abs()
corr_df = corr_matrix.unstack().sort_values(kind='quicksort').reset_index()
corr_df = corr_df[corr_df['level_0'] != corr_df['level_1']].dropna()


# Top 15 most related features
# corr_df.tail(30).sort_values(by=0, ascending=False).drop_duplicates(subset=[0])


# Idea: https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features/

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))
# Find index of feature columns with correlation greater than 0.95
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
# Drop features
df.drop(to_drop, axis = 1, inplace=True)


# Encoding

obj_df = df.select_dtypes(include=['object']).copy()
# obj_df.columns

to_drop = ['match_date','away_team_player10', 'away_team_player11', 'away_team_player2',
       'away_team_player3', 'away_team_player4', 'away_team_player5',
       'away_team_player6', 'away_team_player7', 'away_team_player8',
       'away_team_player9', 'home_team_player1', 'home_team_player10',
       'home_team_player11', 'home_team_player2', 'home_team_player3',
       'home_team_player4', 'home_team_player5', 'home_team_player6',
       'home_team_player7', 'home_team_player8', 'home_team_player9', 'away_team_player1']

df.drop(to_drop, 1, inplace=True)
df.drop(['home_team_score', 'away_team_score', 'attendance_percent'], 1, inplace = True)
obj_df.drop(to_drop, 1, inplace=True)

# obj_df.nunique()
obj_cols = obj_df.columns.tolist()
obj_df = pd.get_dummies(obj_df)
df = df.merge(obj_df, how='left', left_on = df.index, right_on = obj_df.index).drop(obj_cols + ['key_0'], 1)


df.to_csv(path + 'corr_and_encoding.py', index=False)
