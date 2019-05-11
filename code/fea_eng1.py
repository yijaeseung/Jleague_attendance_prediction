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


key = input('Enter your google API key: ')
gmaps = googlemaps.Client(key=key)


def add_datepart(df, fldname, drop=True, time=False):
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start']
    if time: attr = attr + ['Hour']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop: df.drop(fldname, axis=1, inplace=True)


# Dictionaries for unifying name of teams & venues
# - The names of teams and venues' have changed as time have passed. However, it is better to unify name of them for better prediction of model. I found that there are no problem if I unified the name of them which I will explain more later. Hence, I used these dictionaries for unification.
same_stadium_dict = {
                    '日産スタジアム':'横浜国際総合競技場',
                    '静岡スタジアムエコパ':'エコパスタジアム',
                    '大阪長居スタジアム':'ヤンマースタジアム長居',
                    '長居スタジアム':'大阪長居第２陸上競技場',
                    '長居第２陸上競技場':'大阪長居第２陸上競技場',
                    '大阪長居第2陸上競技場':'大阪長居第２陸上競技場',
#                     'デンカビッグスワンスタジアム':'東北電力ビッグスワンスタジアム',
                    '大分銀行ドーム':'大分スポーツ公園総合競技場',
                    '九州石油ドーム':'大分スポーツ公園総合競技場',
                    '市立吹田サッカースタジアム':'パナソニックスタジアム吹田',
                    'エディオンスタジアム広島':'広島ビッグアーチ',
                    '熊本県民総合運動公園陸上競技場':'うまかな・よかなスタジアム',
                    'えがお健康スタジアム':'うまかな・よかなスタジアム',
                    'Ｐｉｋａｒａスタジアム':'香川県立丸亀競技場',
                    '東平尾公園博多の森陸上競技場':'博多の森陸上競技場',
                    'ノエビアスタジアム神戸':'ホームズスタジアム神戸',
                    '草薙総合運動公園陸上競技場':'静岡県草薙総合運動場陸上競技場',
                    '静岡県営草薙陸上競技場':'静岡県草薙総合運動場陸上競技場',
#                     '名古屋市瑞穂陸上競技場':'パロマ瑞穂スタジアム',
#                     '新潟スタジアム': '東北電力ビッグスワンスタジアム',
                    'ShonanBMWスタジアム平塚': 'ＳｈｏｎａｎＢＭＷスタジアム平塚',
                    '瑞穂公園陸上競技場':'パロマ瑞穂スタジアム',
                    '岐阜メモリアルセンター長良川競技場':'岐阜メモリアルセンター長良川球技メドウ',
                    '沖縄県総合運動公園陸上競技場':'高知県立春野総合運動公園陸上競技場',
                    '鳥栖スタジアム':'ベストアメニティスタジアム',
                    'レベルファイブスタジアム':'東平尾公園博多の森球技場',
                    'さいたま市浦和駒場スタジアム':'浦和市駒場スタジアム',
                    '京都市西京極総合運動公園陸上競技場':'京都市西京極総合運動公園陸上競技場兼球技場',
                    'ＩＡＩスタジアム日本平':'日本平スタジアム',
                    'IAIスタジアム日本平': '日本平スタジアム',
                    'アウトソーシングスタジアム日本平':'日本平スタジアム',
                    '日本平運動公園球技場':'日本平スタジアム',
                     '名古屋市瑞穂球技場' : 'パロマ瑞穂スタジアム',
                    'NDソフトスタジアム山形': 'ＮＤソフトスタジアム山形',
                    '山形県総合運動公園陸上競技場':'ＮＤソフトスタジアム山形',
                    'トランスコスモススタジアム長崎':'長崎県立総合運動公園陸上競技場',
                    '駒沢陸上競技場':'駒沢オリンピック公園総合運動場陸上競技場',
                    '松本平広域公園総合球技場':'サンプロアルウィン',
                    '維新百年記念公園陸上競技場':'維新みらいふスタジアム',
                    'ｋａｎｋｏスタジアム':'シティライトスタジアム',
                    '岡山県陸上競技場桃太郎スタジアム':'シティライトスタジアム',
                    'ニンジニアスタジアム':'愛媛県総合運動公園陸上競技場',
                    '仙台スタジアム':'ユアテックスタジアム仙台',
                    '七北田公園仙台スタジアム':'ユアテックスタジアム仙台',
                    '栃木県立グリーンスタジアム':'栃木県グリーンスタジアム',
                    '徳島県鳴門総合運動公園陸上競技場':'鳴門・大塚スポーツパークポカリスエットスタジアム',
                    '長野運動公園総合運動場':'長野市営長野運動公園総合運動場',
                    '山梨中銀スタジアム':'山梨県小瀬スポーツ公園陸上競技場',
                    'とりぎんバードスタジアム':'鳥取市営サッカー場バードスタジアム',
                    '埼玉県営大宮公園サッカー場':'ＮＡＣＫ５スタジアム大宮',
                    'さいたま市大宮公園サッカー場':'ＮＡＣＫ５スタジアム大宮',
                    'NACK5スタジアム大宮': 'ＮＡＣＫ５スタジアム大宮',
                    '三協フロンテア柏スタジアム':'日立柏サッカー場',
                    '正田醤油スタジアム群馬':'群馬県立敷島公園県営陸上競技場',
                    '平塚競技場':'ＳｈｏｎａｎＢＭＷスタジアム平塚',
                    'ジュビロ磐田スタジアム':'ジュビロ磐田サッカースタジアム',
                    'ヤマハスタジアム（磐田）':'ジュビロ磐田サッカースタジアム',
                    'ヤマハスタジアム(磐田)':'ジュビロ磐田サッカースタジアム',
                    '横浜市三ツ沢公園球技場':'ニッパツ三ツ沢球技場',
                    '函館市千代台公園陸上競技場':'千代台公園陸上競技場',
                    '水戸市立競技場':'ケーズデンキスタジアム水戸',
                    '西が丘サッカー場':'国立スポーツ科学センター西が丘サッカー場',
                    '味の素フィールド西が丘':'国立スポーツ科学センター西が丘サッカー場',
                    '国立西が丘サッカー場':'国立スポーツ科学センター西が丘サッカー場', 
                    '埼玉スタジアム２００２': '埼玉スタジアム2002'
                }


def tidying_venue_name(df):
    same_stadium_dict = {
                        '日産スタジアム':'横浜国際総合競技場',
                        '静岡スタジアムエコパ':'エコパスタジアム',
                        '大阪長居スタジアム':'ヤンマースタジアム長居',
                        '長居スタジアム':'大阪長居第２陸上競技場',
                        '長居第２陸上競技場':'大阪長居第２陸上競技場',
                        '大阪長居第2陸上競技場':'大阪長居第２陸上競技場',
#                         'デンカビッグスワンスタジアム':'東北電力ビッグスワンスタジアム',
                        '大分銀行ドーム':'大分スポーツ公園総合競技場',
                        '九州石油ドーム':'大分スポーツ公園総合競技場',
                        '市立吹田サッカースタジアム':'パナソニックスタジアム吹田',
                        'エディオンスタジアム広島':'広島ビッグアーチ',
                        '熊本県民総合運動公園陸上競技場':'うまかな・よかなスタジアム',
                        'えがお健康スタジアム':'うまかな・よかなスタジアム',
                        'Ｐｉｋａｒａスタジアム':'香川県立丸亀競技場',
                        '東平尾公園博多の森陸上競技場':'博多の森陸上競技場',
                        'ノエビアスタジアム神戸':'ホームズスタジアム神戸',
                        '草薙総合運動公園陸上競技場':'静岡県草薙総合運動場陸上競技場',
                        '静岡県営草薙陸上競技場':'静岡県草薙総合運動場陸上競技場',
                        '名古屋市瑞穂陸上競技場':'パロマ瑞穂スタジアム',
#                         '新潟スタジアム': '東北電力ビッグスワンスタジアム',
                        'ShonanBMWスタジアム平塚': 'ＳｈｏｎａｎＢＭＷスタジアム平塚',
                        '瑞穂公園陸上競技場':'パロマ瑞穂スタジアム',
                        '岐阜メモリアルセンター長良川競技場':'岐阜メモリアルセンター長良川球技メドウ',
                        '沖縄県総合運動公園陸上競技場':'高知県立春野総合運動公園陸上競技場',
                        '鳥栖スタジアム':'ベストアメニティスタジアム',
                        'レベルファイブスタジアム':'東平尾公園博多の森球技場',
                        'さいたま市浦和駒場スタジアム':'浦和市駒場スタジアム',
                        '京都市西京極総合運動公園陸上競技場':'京都市西京極総合運動公園陸上競技場兼球技場',
                        'ＩＡＩスタジアム日本平':'日本平スタジアム',
                        'IAIスタジアム日本平': '日本平スタジアム',
                        'アウトソーシングスタジアム日本平':'日本平スタジアム',
                        '日本平運動公園球技場':'日本平スタジアム',
#                          '名古屋市瑞穂球技場' : 'パロマ瑞穂スタジアム',
                        'NDソフトスタジアム山形': 'ＮＤソフトスタジアム山形',
                        '山形県総合運動公園陸上競技場':'ＮＤソフトスタジアム山形',
                        'トランスコスモススタジアム長崎':'長崎県立総合運動公園陸上競技場',
                        '駒沢陸上競技場':'駒沢オリンピック公園総合運動場陸上競技場',
                        '松本平広域公園総合球技場':'サンプロアルウィン',
                        '維新百年記念公園陸上競技場':'維新みらいふスタジアム',
                        'ｋａｎｋｏスタジアム':'シティライトスタジアム',
                        '岡山県陸上競技場桃太郎スタジアム':'シティライトスタジアム',
                        'ニンジニアスタジアム':'愛媛県総合運動公園陸上競技場',
                        '仙台スタジアム':'ユアテックスタジアム仙台',
                        '七北田公園仙台スタジアム':'ユアテックスタジアム仙台',
                        '栃木県立グリーンスタジアム':'栃木県グリーンスタジアム',
                        '徳島県鳴門総合運動公園陸上競技場':'鳴門・大塚スポーツパークポカリスエットスタジアム',
                        '長野運動公園総合運動場':'長野市営長野運動公園総合運動場',
                        '山梨中銀スタジアム':'山梨県小瀬スポーツ公園陸上競技場',
                        'とりぎんバードスタジアム':'鳥取市営サッカー場バードスタジアム',
                        '埼玉県営大宮公園サッカー場':'ＮＡＣＫ５スタジアム大宮',
                        'さいたま市大宮公園サッカー場':'ＮＡＣＫ５スタジアム大宮',
                        'NACK5スタジアム大宮': 'ＮＡＣＫ５スタジアム大宮',
                        '三協フロンテア柏スタジアム':'日立柏サッカー場',
                        '正田醤油スタジアム群馬':'群馬県立敷島公園県営陸上競技場',
                        '平塚競技場':'ＳｈｏｎａｎＢＭＷスタジアム平塚',
                        'ジュビロ磐田スタジアム':'ジュビロ磐田サッカースタジアム',
                        'ヤマハスタジアム（磐田）':'ジュビロ磐田サッカースタジアム',
                        'ヤマハスタジアム(磐田)':'ジュビロ磐田サッカースタジアム',
                        '横浜市三ツ沢公園球技場':'ニッパツ三ツ沢球技場',
                        '函館市千代台公園陸上競技場':'千代台公園陸上競技場',
                        '水戸市立競技場':'ケーズデンキスタジアム水戸',
                        '西が丘サッカー場':'国立スポーツ科学センター西が丘サッカー場',
                        '味の素フィールド西が丘':'国立スポーツ科学センター西が丘サッカー場',
                        '国立西が丘サッカー場':'国立スポーツ科学センター西が丘サッカー場', 
                        '埼玉スタジアム２００２': '埼玉スタジアム2002'
                    }
    
    df['venue'] = df['venue'].replace(to_replace= same_stadium_dict)





same_team_dict = {
         'C大阪':'Ｃ大阪',
         'G大阪':'Ｇ大阪',
         '川崎F':'川崎Ｆ',
         'FC東京':'FC東京',
         '東京Ｖ':'東京Ｖ',
         '横浜FC':'横浜FC',
         '横浜FM':'横浜FM',
         '横浜FM-F':'横浜Ｆ',
         '横浜FM-M':'横浜M',
          'Ｖ川崎': '東京Ｖ',
          'Ｆ東京': 'FC東京',
          '草津': '群馬',
          '平塚' : '湘南'
        }



def tidying_team_name(df):
    same_team_dict = {'C大阪':'Ｃ大阪','G大阪':'Ｇ大阪','川崎F':'川崎Ｆ','FC東京':'FC東京','東京Ｖ':'東京Ｖ','横浜FC':'横浜FC',
             '横浜FM':'横浜FM','横浜FM-F':'横浜Ｆ','横浜FM-M':'横浜M','Ｖ川崎': '東京Ｖ','Ｆ東京': 'FC東京','草津': '群馬',
              '東京V': '東京Ｖ','平塚' : '湘南','市原': '千葉'
            }
    
    df['home_team']= df['home_team'].replace(to_replace=same_team_dict)
    df['away_team']= df['away_team'].replace(to_replace=same_team_dict)




# Load Data
IS_LOCAL = True

if (IS_LOCAL):
    path = './00_Data/'
    
else:
    path = '../input/'
    
os.listdir(path)


# - The datasets given by competition are about J1-league matches of 2006 ~ 2018. However, we can access and use J1 & J2 league data of 1993 to 2018 from official J-league website. Hence, I used those extended datasets as basic datasets.



train_df = pd.read_csv(path + 'ex_total.csv')
train_df = train_df.loc[train_df['match_date'] < '2017-01-01']

test_df = pd.read_csv(path + 'test.csv')
test_ids = test_df['id'].values

cap_df = pd.read_csv(path + 'stadium_capacity_mapping.csv')
ex_cap_df = pd.read_csv(path + 'ex_stadium_capacity_mapping.csv')

reports_df = pd.read_csv(path + 'ex_match_reports.csv')
reports_df = reports_df.loc[(reports_df['id'].isin(train_df['id'].values)) | (reports_df['id'].isin(test_ids))]")


# Lots of capacity data are missing on ex_cap_df. I used capacity dataset given by competition and external data to fill in those values.


cap = pd.read_csv(path + 'stadium_capacity_mapping.csv')
cap.rename(columns={'stadium': 'venue'}, inplace=True)
ex_cap_df.rename(columns={'stadium': 'venue'}, inplace=True)

common_venues = []
for x in ex_cap_df.venue.unique():
    if x in cap.venue.unique():
        common_venues.append(x)
        
venues_to_fill = ex_cap_df.loc[(ex_cap_df.venue.isin(common_venues))&(ex_cap_df.capacity.isnull())].venue.unique()
ex_cap_df.loc[ex_cap_df.venue.isin(venues_to_fill), 'capacity'] = ex_cap_df.loc[ex_cap_df.venue.isin(venues_to_fill), 'venue'].apply(lambda x: cap.loc[cap.venue == x, 'capacity'].values[0])

capacities_for_null = [15454, 35000, 28000, 27495, 14051, 15353, 12000, 9245, 21053, 10081, 47000, 20125, 47000, 19694, 30000, 20010, 28000, 49970, 30132,
                24130, 5000, 17200, 5500, 11105, 17200, 15000, 10000, 7000, 5000, 7500, 20000, 47000, 26109, 4300]

ex_cap_df.loc[ex_cap_df.capacity.isnull(), 'capacity'] = capacities_for_null
tidying_venue_name(ex_cap_df)

# Fixed the capacity of '宮城陸上競技場'
ex_cap_df.loc[ex_cap_df.venue == '宮城陸上競技場', 'capacity'] = 30000

# Some venues appear multiple times after revising names of venues, and they have same capacities. Drop that duplicates.
cap_dupl_check = ex_cap_df.venue.value_counts()
not_unique_venues = cap_dupl_check[cap_dupl_check >= 2].index
ex_cap_df[ex_cap_df.venue.isin(not_unique_venues)].sort_values(by='venue')
ex_cap_df = ex_cap_df.drop_duplicates()



# Feature Engineering

# 1) Merge Capacity Feature to Dataset

df = pd.concat([train_df,test_df])
df['id'] = pd.to_numeric(df['id'], errors='ignore')




df['division'] = df['division'].fillna(1)
df['home_team']= df['home_team'].replace(to_replace=same_team_dict)
df['away_team']= df['away_team'].replace(to_replace=same_team_dict)
tidying_venue_name(df)


df = df.merge(right = ex_cap_df, how='left', on = 'venue')

df = df.sort_values(by='id')


# Clean round and kick_off_time feature




df['round'] = df['round'].apply(lambda x: re.search('\d+', x)[0])
# Unify style of number.
df['round'] = df['round'].apply(lambda x: jaconv.z2h(x, digit=True))
df['round'] = pd.to_numeric(df['round'], errors = 'coerce')


# - Converted kick off time to seconds to use the feature in the model.



def get_seconds(data):
    h, m, s = data.split(':')
    seconds = int(h) * 3600 + int(m) * 60 + int(s)
    
    return seconds


df['kick_off_time'] = pd.to_datetime(df['kick_off_time'], errors='ignore')
df['kick_off_time'] = df['kick_off_time'].apply(lambda x: str(x.time()))
df['kick_off_time'] = df['kick_off_time'].apply(lambda x: get_seconds(x))


# 2) Datetime Features 



add_datepart(df, 'match_date', drop=False)
df[['match_Is_month_end','match_Is_month_start','match_Is_quarter_end','match_Is_quarter_start']] = df[['match_Is_month_end','match_Is_month_start','match_Is_quarter_end','match_Is_quarter_start']].astype(np.int8)

# Added attendance / capacity since it can show the absolute popularity about the game.
df['attendance_percent'] = df['attendance'] / df['capacity']


# 3) Modify Section in Chronological Order

# - In some seasons, sections given to each match are not ordered in time sequence. For example, some matches supposed to be held in march and april of 2011 are postponed because of earthquake in Hukushima prefecture. However, the sections that are given to matches wasn't changed.
# - It is better to revise section chronologically, because it allows us more information about each match. For example, win ratio calculated based on original section can't give the accurate information about how each team performs recently, since section is not chronological. Morever, we can easily engineer features like current ranking or win in a row of each team by revising section.

def revise_section(row, status):
    id_ = row['id']
    team = row['{}_team'.format(status)]
    new_section = section_df.loc[(section_df['status'] == '{}_team'.format(status))&(section_df['id'] == id_), 'section'].values[0]
    return new_section

for year in range(1993,2019):
    for division in [1,2]:
        temp = df.loc[(df['match_Year'] == year)&(df['division'] == division)].sort_values(by='match_date')[['id', 'match_date', 'home_team', 'away_team']]
        temp = pd.melt(temp, id_vars=['id', 'match_date'], value_vars=['home_team', 'away_team'], var_name= 'status', value_name = 'team')
        section_df = pd.DataFrame(columns=list(temp.columns) + ['section'])
        teams = temp['team'].unique()
        ids = temp['id'].values
        
        for team in teams:
            team_df = temp.loc[temp['team'] == team].sort_values(by='match_date')
            team_df['section'] = team_df.reset_index().index.values + 1
            section_df = pd.concat([section_df, team_df])
        for status in ['home', 'away']:
            df.loc[df['id'].isin(temp['id'].values), '{}_section'.format(status)] = df.loc[df['id'].isin(temp['id'].values)].apply(revise_section, axis=1, status=status)


# 4) Previous Division Feature

# - I will make the feature about in which division each team was belonged in previous season. we can expect positive effect on the number of audience if home team promoted from J2 league last year. On the other hand, we can expect some negative effect if away team is just promoted, since they might be less popular than other teams stayed at J1 league for long time. 

# - 1: Division 1
# - 2: Division 2
# - 3: Division 3
# - 4: Allocated to teams in 1993, when J1 league is started.

def previous_division(row, df):
    year = row['match_Year']
    division = row['division']
    
    try:
        previous_year = year-1
        previous_division = df.loc[df['match_Year'] == previous_year, 'division'].values[0]
        return previous_division
    
    except:
        if year == 1993:
            return 4
        elif division == 2:
            return 3
        elif year < 2000 and division == 1:
            return 2
        elif year == 2018 and division == 1:
            return 2
        else:
            return np.nan


league_df = df[['home_team', 'match_Year', 'division']].drop_duplicates().rename(columns={'home_team': 'team'})
teams = league_df.team.unique()
division_df = pd.DataFrame(columns = list(league_df.columns) + ['previous_division'])

for team in teams:
    temp = league_df.loc[league_df['team'] == team]
    temp['previous_division'] = temp.apply(previous_division, axis=1, df=temp)
    division_df = pd.concat([division_df, temp])
    
division_df['match_Year'] = division_df['match_Year'].astype('int')

df = df.merge(right=division_df[['match_Year', 'team', 'previous_division']], left_on = ['match_Year', 'home_team'], right_on = ['match_Year', 'team']).drop(['team'], 1).rename(columns={'previous_division': 'home_prev_div'})
df = df.merge(right=division_df[['match_Year', 'team', 'previous_division']], left_on = ['match_Year', 'away_team'], right_on = ['match_Year', 'team']).drop(['team'], 1).rename(columns={'previous_division': 'away_prev_div'})


# 5) Process 'weather' Feature

# - I needed to process weather, the daily weather forecast, to let it be more useful to predict audiences. 
# - I divided daily weather forecast to 3 time zones. (07~12, 12~18, 18~24)
# - This will help model to capture the effect of model than before.
# - For example, if kick off time is 17:00 pm, we can guess that weather of 12~18pm might be the important factor when deciding whether to go to see the match. Model can capture this effect easily by dividing weather information into each timezone.


weather_df = df.loc[df['weather'].notnull(), ['id', 'weather']]
weather_index = weather_df.index

wea_daily = weather_df['weather'].tolist()
wea_daily = [x.split('のち') for x in wea_daily]

wea_timezone = []

for x in wea_daily:
    if len(x) == 1:
        result = [x[0], x[0], x[0]]
        wea_timezone.append(result)   

    elif len(x) == 2:
        result = [x[0], x[1], x[1]]
        wea_timezone.append(result)

    elif len(x) == 3:
        wea_timezone.append(x)



wea_0712 = [x[0] for x in wea_timezone]
wea_1218 = [x[1] for x in wea_timezone]
wea_1824 = [x[2] for x in wea_timezone]

weather_df['weather_0712'] = wea_0712
weather_df['weather_1218'] = wea_1218
weather_df['weather_1824'] = wea_1824

df = df.merge(right=weather_df, how='left', on = ['id', 'weather'])
df.drop(['weather'], 1, inplace=True)


# 6) Process 'broadcasters' Feature

# - The feature shows the broadcasters of each match, divided by '/'. I will count the number of broadcasters to make new feature.
# - However, I will not count some broadcasters for better prediction.
# Skyper TV and DAZN are broadcasters that have(had) the right to broadcast all J league games.
# It would not be any problem if they have existed for whole period of dataset. However, they existed only for certain period.
# This means 1 will be added to the count of broadcasters for all games in that period, even they can't explain people's interest about that game
# because those companies broadcast every game of season. Hence, it is better to exlude these broadcasters from counting.

# - Japanese uses halfwidth and fullwidth forms of weather. I unificated all letters to halfwidths form for better comparison.


df['broad_list'] = df['broadcasters'].str.split('/')

for i, row in df.iterrows():
    broad_list = row['broad_list']
    try: 
        df.loc[df.index == i, 'broad_list'] = df.loc[df.index == i, 'broad_list'].apply(lambda x: [a for a in x if 'スカ' not in a])
        df.loc[df.index == i, 'broad_list'] = df.loc[df.index == i, 'broad_list'].apply(lambda x: [a for a in x if 'DAZN' not in a])
        df.loc[df.index == i, 'broad_list'] = df.loc[df.index == i, 'broad_list'].apply(lambda x: [a for a in x if 'ＤＡＺＮ' not in a])
    
    except:
        df.loc[df.index == i, 'broad_list'] = np.nan
        
df.loc[df.broad_list.notnull(), 'broad_num'] = df.loc[df.broad_list.notnull(), 'broad_list'].apply(lambda x: len(x))
del df['broad_list'], df['broadcasters']


# 7) Humidex Feature

# - I added humidex feature using 'temparture' and 'humidity' features which are given by dataset.
# I expected humidex can reflect the impact of weather better than just using original features, since this is the index about how the people feel about the weather.

def cal_humidex(row):
    
    def cal_dew_point(temp, humidity):         
        temp = Temp(temp, 'c')
        dewpoint = dew_point(temperature=temp, humidity = humidity).c
        
        return dewpoint
 

    temp = row['temperature']
    humidity = row['humidity']    
    if (np.isnan(temp) == False) & (np.isnan(humidity) == False):
        if temp < 15:
            return 0
        
        else:            
            dewpoint = cal_dew_point(temp, humidity)      
            humidex = temp + 0.5555 * ( 6.11 * math.e ** round( 5417.7530*( (1/273.16) - (1/(273.15 + dewpoint) ) ), 5)- 10 )

            return humidex
    
    else:
        return np.nan
    
df['humidex'] = df.apply(cal_humidex, axis=1)


# 8) Season Performance Features

# - I guessed that the seasonal performance of team like current ranking, recent win ratio or final ranking of last season might affect the attendance. Hence, I made that features using match reports dataset.

# - Tree base model can't use data of other rows to get meaningful information, like win ratio for recent n games. Hence, I calculated it.
# - Win: 1, Tie: 0.5, Lose: 0
# - Generally, The number of wins only matters when calculating win ratio. However, it is quite different whether a team resulted 2W/3T or 2W/3L. Hence, I gave 0.5 to tie match.
# - I calculated win ratios of recent 5 games and of 9 games. I will test them and choose which one I will use for prediction.


df = df.merge(reports_df, how = 'left', on = 'id')
ratio_df = df[['id', 'match_Year', 'match_date', 'division', 'home_team', 'away_team', 'home_team_score', 'away_team_score']]

ratio_df = pd.melt(frame=ratio_df, id_vars=['id', 'match_Year', 'match_date', 'division', 'home_team_score', 'away_team_score'],
       value_vars = ['home_team', 'away_team'], var_name='status', value_name='team')


# - I calculated result of each match using 'home_team_score' and 'away_team_score' features.

# 1 if home_team_score > away_team_score, 0 if home_team_score = away_team_score, -1 if home_team_score < away_team_score
ratio_df.loc[(ratio_df['home_team_score'] > ratio_df['away_team_score']), 'result'] = 1
ratio_df.loc[(ratio_df['home_team_score'] == ratio_df['away_team_score']), 'result'] = 0
ratio_df.loc[(ratio_df['home_team_score'] < ratio_df['away_team_score']), 'result'] = -1

# Put (-) for away teams, since win of the home team means lose of the away team. 
ratio_df.loc[(ratio_df['status'] == 'away_team'), 'result'] = ratio_df.loc[(ratio_df['status'] == 'away_team'), 'result'].apply(lambda x: -x)

# Changed the value of the no decision and lose for calculation of win ratio.
ratio_df.loc[ratio_df['result'] == 0, 'result'] = 0.5
ratio_df.loc[ratio_df['result'] == -1, 'result']= 0

ratio_df = ratio_df.drop(['home_team_score', 'away_team_score'], 1)

perform_df = pd.DataFrame(columns = ratio_df.columns.tolist() + ['win_ratio_5games', 'win_ratio_9games', 'season_points', 'section'] )

def cal_points(result):
    if result == 1:
        return 3
    if result == 0.5:
        return 1
    if result == 0:
        return 0


# - Caculated the win ratio for recent 5 & 9 games using match results.
years = range(1993, 2019)

for year in years:
    teams = ratio_df['team'].unique()
    for team in teams:
        temp = ratio_df.loc[(ratio_df['team'] == team)&(ratio_df['match_Year'] == year)].sort_values(by='match_date').reset_index().drop(['index'], 1)
        
        temp['season_points'] = temp['result'].apply(cal_points).cumsum()        
        temp['section'] = temp.index.astype(int) + 1
        
        for i, row in temp.iterrows():
            for num in [5, 9]:
                if i < num: 
                    continue
                else:
                    matches = range(i-num, i)
                    result = sum(temp.iloc[matches]['result'].values.tolist())
                    temp.loc[temp.index == i, 'win_ratio_{}games'.format(num)] = result
           
        perform_df = pd.concat([perform_df, temp], ignore_index=True)


for col in ['section', 'season_points']:    
    perform_df[col] = pd.to_numeric(perform_df[col], errors = 'coerce')


# - Calculated current ranking
years = range(1993, 2019)

for year in years:
    for division in [1,2]:
        try:
            
            if year == 2018 and division == 1:
                temp = perform_df.loc[(perform_df['division'] == division) & (perform_df['match_Year'] == year) & (perform_df['section'] < 18), ['section', 'team', 'season_points']]
            else:
                temp = perform_df.loc[(perform_df['division'] == division) & (perform_df['match_Year'] == year), ['section', 'team', 'season_points']]

            ranking = pd.pivot_table(data=temp, index=['section'], columns = ['team'], values = ['season_points'])
            ranking = ranking.rank(axis=1, method='min', ascending=False)['season_points'].reset_index()

            sections = ranking['section'].unique().tolist()
            teams = ranking.columns.tolist()[1:]      # Added the result of match to the next match.

            for section in sections:
                for team in teams:
                    perform_df.loc[(perform_df['team'] == team) & (perform_df['match_Year'] == year) & (perform_df['section'] == section), 'ranking'] =                     ranking[team].loc[ranking['section'] == section].values[0]
        
        except:
            continue


# - Use current ranking of last match for each season as a final ranking for the season
year_list = []
team_list = []
ranking_list = []

for year in range(1993, 2019):
    teams = perform_df.loc[perform_df['match_Year'] == year, 'team'].unique()
    for team in teams:
        temp = perform_df.loc[(perform_df['match_Year'] == year) & (perform_df['team'] == team)]
        if year != 2018:
            season_ranking = temp['ranking'].values[-1]
            ranking_list.append(season_ranking)
            year_list.append(year)
            team_list.append(team)
        else:
            continue
        
        perform_df.loc[(perform_df['match_Year'] == year) & (perform_df['team'] == team), 'current_ranking'] =         temp['ranking'].shift(1)

final_rank_df = pd.DataFrame(data=dict(zip(['year', 'team', 'last_season_ranking'], [year_list, team_list, ranking_list])))

# Added the final ranking of season to next season.
final_rank_df['year'] += 1

merge_df = perform_df[['id', 'status', 'win_ratio_5games', 'win_ratio_9games', 'current_ranking']]
merge_df['id'] = merge_df['id'].astype(int)
merge_home = merge_df.loc[merge_df['status'] == 'home_team'].drop(['status'], 1)
merge_away = merge_df.loc[merge_df['status'] == 'away_team'].drop(['status'], 1)

merge_cols = ['win_ratio_5games', 'win_ratio_9games', 'current_ranking']

df = df.merge(right=merge_home, how='left', on = ['id']).rename(columns=dict(zip(merge_cols, ['home_' + x for x in merge_cols])))
df = df.merge(right=merge_away, how='left', on = ['id']).rename(columns=dict(zip(merge_cols, ['away_' + x for x in merge_cols]))).sort_values(by='id')

for status in ['home', 'away']:
    df = df.merge(right=final_rank_df, how='left', left_on = ['match_Year', '{}_team'.format(status)], right_on = ['year', 'team']).    drop(['year', 'team'], 1).rename(columns={'last_season_ranking': '{}_last_season_ranking'.format(status)})


# 10) Time Lag Features

# - Past attendances data is valuable information for predicting future attendances. Hence, I calculated time-lagged average attendances in various conditions.

# - Used ratio (Attendance / Capacity) instead of attendance. It's because the value of attendance itself can't reflect well about the popularity of the match since attendance is limited by capacity.
# - Calculated following features for recent 1~4 years.

# 1. Yearly average of attendance for the venue
# 2. Yearly average of attendance for home & away team
# 3. Monthly average of attendance for away & away team

def cal_lag_attendance(df):
    
        
    for year in range(1993, 2019):
        
        unique_team = df.loc[df['match_Year'] == year, 'home_team'].unique()
        unique_venue = df.loc[df['match_Year'] == year, 'venue'].unique()     

        for year_, team, venue in product([str(year)], unique_team, unique_venue):
            for lag in range(1, 5):
                for loc in ['home', 'away']:
                    temp = df[(df['match_Year'] == int(year_) - lag) & (df['{}_team'.format(loc)] == team) & (df['venue'] == venue)]

                    if len(temp) != 0:
                        mean_ = temp['attendance_percent'].mean()
                        df.loc[(df['match_Year'] == int(year_)) & (df['{}_team'.format(loc)] == team) & (df['venue'] == venue),'{}_avg_attendance_by_venue_lag{}'.format(loc, lag)] = mean_

    for year in range(1993, 2019):    
        
        for year_, month, team in product([str(year)], [str(x) for x in range(2, 13)],df.loc[df['match_Year'] == year,'home_team'].unique()):            
            for loc in ['home', 'away']:
                for lag in range(1, 5):
                    temp1 = df[(df['match_Year'] == int(year_) - lag)& (df['match_Month'] == int(month))& (df['{}_team'.format(loc)] == team)]
                    if len(temp1) != 0:
                        mean_ = temp1['attendance_percent'].mean()
                        df.loc[(df['match_Year'] == int(year_)) & (df['match_Month'] == int(month)) & (df['{}_team'.format(loc)] == team),'{}_avg_attendance_by_month_lag{}'.format(loc, lag)] = mean_
                    
                    temp2 = df[(df['match_Year'] == int(year_) - lag) & (df['{}_team'.format(loc)] == team)]
                    if len(temp2) != 0:
                        mean_ = temp2['attendance_percent'].mean()
                        df.loc[(df['match_Year'] == int(year_)) & (df['{}_team'.format(loc)] == team), '{}_avg_attendance_by_year_lag{}'.format(loc, lag)] = mean_
    
    return df

df = cal_lag_attendance(df)

df.to_csv(path + 'fea_eng1.csv', index=False)
