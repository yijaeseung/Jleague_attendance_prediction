
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


df = pd.read_csv(path + 'fea_eng1.py')


# 5-2. Add New Features using External Data

# 1) Merge Area Name and Code to Venue, Home, and Away Team
# - The information about in which area team or venue is located in can help model to predict the attendance both in direct and indirect way.
# It can give some fundamental information about the area, like population of the area.
# It also can help to make region based features like distance between the base of the teams.

# Venue
venues = df['venue'].dropna().unique()
venue_areas = []

for venue in venues:
    raw_data = gmaps.geocode(address = venue, language = 'jp')
    
    venue_area = raw_data[0]['formatted_address']
    venue_areas.append(venue_area)

venue_area_df = pd.DataFrame(data={'venue': venues, 'venue_area': venue_areas})


def extract_area_name(row):
    add1 = re.compile('〒\d{3}-\d{4} (\w+(-ken|県$| *\w+))')
    add2 = re.compile('(\w+ *\w*) \d{3}-\d{4}')
    add3 = re.compile('(\w+ *\w*), \d{3}-\d{4}')
    
    address = row['venue_area']
    
    try:
        area_name1 = add1.search(address).groups(0)[0]
        try:           
            area_name1_done = re.search('\w+(県)',area_name1)[0]
            return area_name1_done
        
        except:
            return area_name1
        
    except:
        try:
            area_name2 = add2.search(address).groups(0)[0]
            return area_name2
        
        except:
            area_name3 = add3.search(address).groups(0)[0]
            return area_name3

pref_code = pd.read_csv(path + 'prefecture_code.csv')
pref_code= pref_code[['AREA', 'AREA Code']].drop_duplicates()
pref_code.columns = ['area', 'area_code']

venue_area_df['venue_area'] = venue_area_df.apply(extract_area_name, axis=1)

# Cleaning area name got from google api to fit in the format of prefecture_code.csv
def tidying_area_name(row):
    kanji_name = ['愛知県', '三重県', '富山市 富山県', '福島県', '兵庫県','宮城県']
    translated_name = ['Aichi-ken', 'Mie-ken', 'Toyama-ken', 'Fukushima-ken', 'Hyogo-ken','Miyagi-ken']
    kanji_converter = dict(zip(kanji_name, translated_name))
    
    name = row['venue_area']
    
    if name == 'Osaka':
        return 'Osaka-fu'
    elif name == 'Gunma':
        return 'Gumma-ken'
    elif name == 'Kyoto':
        return 'Kyoto-fu'
    elif name == 'Osaka Prefecture':
        return 'Osaka-fu'
    elif name == 'Ōita Prefecture':
        return 'Oita-ken'
    elif name == 'Kyoto Prefecture':
        return 'Kyoto-fu'
    elif name == 'Hokkaido':
        return 'Hokkaido'
    elif name == 'Hyōgo Prefecture':
        return 'Hyogo-ken'
    elif name == 'Kōchi Prefecture':
        return 'Kochi-ken'
    elif name == 'Gunma Prefecture':
        return 'Gumma-ken'
    elif name == 'Tokyo':
        return 'Tokyo-to'
    elif name in kanji_name:
        return kanji_converter[name]    
    else: 
        return name.split(' ')[0] + '-ken'

venue_area_df['venue_area'] = venue_area_df.apply(tidying_area_name, axis=1)

venue_area_df = venue_area_df.merge(pref_code, how='left', left_on = 'venue_area', right_on = 'area')
del venue_area_df['area']
venue_area_df.rename(columns={'area_code': 'venue_area_code'}, inplace=True)
df = df.merge(right = venue_area_df, how='left', on=['venue'])


# Home & Away Team

team_area_code = pd.read_csv(path + 'football_team_loc_code.csv')

same_team_dict = {'C大阪':'Ｃ大阪','G大阪':'Ｇ大阪','川崎F':'川崎Ｆ','FC東京':'FC東京','東京Ｖ':'東京Ｖ','横浜FC':'横浜FC',
         '横浜FM':'横浜FM','横浜FM-F':'横浜Ｆ','横浜FM-M':'横浜M','Ｖ川崎': '東京Ｖ','Ｆ東京': 'FC東京','草津': '群馬',
          '東京V': '東京Ｖ','平塚' : '湘南','市原': '千葉'}

team_area_code['team'] = team_area_code['team'].replace(to_replace=same_team_dict)
team_area_code = team_area_code.drop_duplicates()

df = df.merge(right=team_area_code, how='left', left_on=['home_team'], right_on=['team']).drop(['team'], 1).rename(columns={'code':'home_team_area_code', 'area': 'home_team_area'})
df = df.merge(right=team_area_code, how='left', left_on=['away_team'], right_on=['team']).drop(['team'], 1).rename(columns={'code':'away_team_area_code', 'area': 'away_team_area'})


# 2) Add Distance, Duration between Teams and Location of each Team

# The Distance and Duration between Teams

# - The reason I add distance & duration between home and away team is because I thought that if the away team locate closer to home team, there will be more fan of away team to come. 
# - However, matches of some home teams are hold in various venues, which sometimes are located in different prefectures. Hence, I calculated distance(duration) using the location of venue that match is hold and prefecture that away team is based on (The location data is derived from googlemaps API).

unique_matches = df.loc[df.venue.notnull(), ['venue', 'away_team_area']].drop_duplicates()

departures = unique_matches.away_team_area.values
arrivals = unique_matches.venue.values

durations = []
distances = []

for dep, arr in zip(departures, arrivals):
    
    # Added this code because google give the location of a restaurant located in Hawaii.
    if dep == 'Chiba-ken':
        dep = '千葉県'
        
    raw_data = gmaps.distance_matrix(dep, arr, language='jp')
    
    try:                 
        dist = raw_data['rows'][0]['elements'][0]['distance']['value'] / 1000
        distances.append(dist)
        duration = float(raw_data['rows'][0]['elements'][0]['duration']['value']) / 60 
        
        if type(duration) == float:
            durations.append(duration)
        else:
            durations.append(NaN)
    except:
#         print(raw_data)
        distances.append(np.nan)
        durations.append(np.nan)

duration_distance_df = pd.DataFrame(data = {'venue': arrivals, 'away_team_area': departures, 'duration': durations, 'distance': distances})
df = df.merge(right=duration_distance_df, on = ['venue', 'away_team_area']).rename(columns={'duration': 'duration_bet_teams', 'distance': 'distance_bet_teams'})


# Location: Latitude and Longitude

def find_lat(row, var_name, key):
    
    stadium = row[var_name]
    
    # read the raw data using geocoding api, provided by Google
    gmaps = googlemaps.Client(key=key)
    raw_data = gmaps.geocode(address=stadium, language='jp')[0]
    
    #Extracting latitude data and adding it to dataframe
    lat = raw_data['geometry']['location']['lat']
    return lat

def find_lon(row, var_name, key):
    
    stadium = row[var_name]
    
    # read the raw data using geocoding api, provided by Google
    gmaps = googlemaps.Client(key=key)
    raw_data = gmaps.geocode(address=stadium, language='jp')[0]
    
    #Extracting latitude data and adding it to dataframe
    global lon
    lon = raw_data['geometry']['location']['lng']
    return lon


venue_locations = pd.DataFrame(data={'venue': df['venue'].values, 'lat': np.nan, 'lon': np.nan}).drop_duplicates()
venue_locations['lat'] = venue_locations.apply(find_lat, axis=1, var_name='venue', key = key)
venue_locations['lon'] = venue_locations.apply(find_lon, axis=1, var_name='venue', key = key)


df = df.merge(right = venue_locations, how='left', on ='venue')


fig, ax = plt.subplots(1, 2, figsize=(10, 4))
i = 0
for col in ['lat', 'lon']:
    i += 1
    plt.subplot(1, 2, i)
    sns.kdeplot(df.loc[df['division'] == 1, col].dropna(), label = 'J1')
    sns.kdeplot(df.loc[df['division'] == 2, col].dropna(), label = 'J2')
    plt.xlabel(col.capitalize())
    plt.title('Distribution of the {}'.format(col.capitalize()))
    plt.legend()

plt.tight_layout()
plt.show()


# 3) Holiday related Features

holiday_df = pd.read_excel(path + 'holiday_extra.xls', sheet_name = '振替休日あり')
holiday_df = holiday_df[['年月日','祝日名']]
holiday_df.columns = ['date', 'description']
holiday_df = holiday_df[holiday_df['date'] > '1992-12-21']

date_df = pd.date_range(start='1992-12-21', end='2018-12-3', freq='D')
date_df = pd.DataFrame(date_df, columns=['date'])

date_df['wod'] = date_df['date'].dt.weekday 
date_df['is_weekend'] = (date_df['wod'] >= 5).astype(np.int8)

date_df = date_df.merge(holiday_df, how='left' , on='date')

date_df['is_holiday'] = date_df['description'].notnull().astype(np.int8)
date_df['is_dayoff'] = date_df['is_weekend'] + date_df['is_holiday']


# - Add how many days left to closest dayoff as feature.
# I added this feature because even the match day is not holiday or weekend (of which the impact is already captured by features made above),
# the number of audiences can show some pattern for some days (ex: the day before dayoff)
date_df = date_df.sort_values('date', ascending=False)
day = np.timedelta64(1, 'D')
closest_dayoff = np.datetime64()
days_to_closest_dayoff = []

for val, current_date in zip(date_df['is_dayoff'].values, date_df['date'].values):
    
    if val != 0:
        closest_dayoff = current_date
        
    days_to_closest_dayoff.append(((closest_dayoff - current_date).astype('timedelta64[D]') / day))

date_df['days_to_closest_dayoff'] = days_to_closest_dayoff
date_df = date_df[['date', 'description', 'is_holiday', 'days_to_closest_dayoff']].rename(columns={'description': 'holiday'})


df = df.merge(right=date_df, how='left', left_on='match_date', right_on='date').drop(['date'], 1)


# 4) Derby Feature

# - Feature about derby matches might help prediction. This can be captured by tree based model by its mechanism, using area code of two teams. However it sometimes fails to capture the effect if I set feature_fraction < 1. Hence, I added this feature. 
# Derby Match:  Match of Teams with Same Regional Base

df['derby'] = 0

df.loc[df['home_team_area_code'] == df['away_team_area_code'], 'derby'] = 1


# 5) Salary related Features

# - Salary can be good feature for measuring team's popularity, which must can be related to the number of attendance.
# - For example, we can expect that famous player like Andreas Iniesta, the former barcelona midfielder who is playing at nissel Kobe now, will let more people visit to the stadium than usual. The highest salary for each team can capture this kind of impact, since star players earn abnormally high salary than normal players.


# Crawl Data

# - Crawled salary data from two websites.
# - extract_salary1: Crawling data from 2002 to 2013
# - extract_salary2: Crawling data from 2013 to 2018
def extract_salary1(year):
    
    player, position, age, salary, team = [], [], [], [], []
    year = str(year)
    
    url = 'http://jsalary.wiki.fc2.com/wiki/%E2%96%A0{}%E5%B9%B4%E2%96%A0'.format(year)
    request = requests.get(url)
    html = request.text
    soup = BS(html)
    
    for x in soup.find_all('a'):
        info = x.get('title')
        
        try:
            if info.startswith('20'):
                team_name = info.split('年')[1]
                team_url = 'http://jsalary.wiki.fc2.com/wiki/{}'.format(info)
                request_t = requests.get(team_url)
                html_t = request_t.text
                soup_t = BS(html_t)
                
                for i, val in enumerate(soup_t.find_all('td')):
                    
                    if i % 4 == 0:
                        player.append(val.text)
                    elif i % 4 == 1:
                        position.append(val.text)
                    elif i % 4 == 2:
                        age.append(val.text)
                    elif i % 4 == 3:
                        salary.append(val.text)
                        team.append(team_name)
                        
        except:
            continue
            
    salary_df = pd.DataFrame(data={'year': year, 'team': team, 'player': player,
                            'salary': salary, 'position': position,
                            'age': age})
    salary_df = salary_df.loc[(salary_df['age'] != '年齢') & (salary_df['position'] != 'ポジション')]
    
    return salary_df





salary_until_2013 = []

for year in range(2002, 2014):
    temp = extract_salary1(year)
    salary_until_2013.append(temp)
    
for i, df_ in enumerate(salary_until_2013):
    if i == 0:
        salary1 = df_
        
    else:
        salary1 = pd.concat([salary1, df_])




def extract_salary2(year):
    if year == 2019:
        url = 'https://www.soccer-money.net/players/in_players.php'
        year = str(year)
        
    else:       
        year = str(year)
        url = 'https://www.soccer-money.net/players/past_in_players.php?year={}'.format(year)

    r = requests.get(url)
    r.encoding = 'utf-8'
    html_doc = r.text
    soup = BS(html_doc)

    ranking = []
    player = []
    age = []
    position = []
    team = []
    salary = []

    for i,x in enumerate(soup.find_all('td')):
        info = x.text
        if i == 0 or i == 1:
            continue
        else:
            crt = i - 2
            if crt % 6 == 0:
                ranking.append(info)
            elif crt % 6 == 1:
                player.append(info)
            if crt % 6 == 2:
                age.append(info)
            if crt % 6 == 3:
                position.append(info)
            if crt % 6 == 4:
                team.append(info)
            if crt % 6 == 5:
                salary.append(info)


    salary_df_ = pd.DataFrame(data={'year': year, 'team': team, 'player': player,
                                    'salary': salary, 'position': position,
                                    'age': age, 'ranking': ranking})
    
    return salary_df_




salary_from_2013 = []

for year in range(2013,2019):
    temp = extract_salary2(year)
    salary_from_2013.append(temp)
    
for i, df_ in enumerate(salary_from_2013):
    if i == 0:
        salary2 = df_
    else:
        salary2 = pd.concat([salary2, df_])


# Clean Crawled Data

# 2002~2013

sal1 = salary1.copy()
sal1 = sal1.reset_index().drop(['index'], 1)

for i, row in sal1.iterrows():
    salary = row['salary']
    if '年俸記載なし' in salary:
        try:
            sal1.loc[sal1.index == i, 'salary'] = re.search('(\d+、*\d+)万',salary).groups(0)[0]
        
        except:
            continue


# - Change teams' names in crawled data to fit with main dataset
team_name_before = ['コンサドーレ札幌', 'ベガルタ仙台', '鹿島アントラーズ', '浦和レッドダイヤモンズ', 'ジェフユナイテッド市原',
       '柏レイソル', 'FC東京', '東京ヴェルディ', '横浜F・マリノス', '清水エスパルス', 'ジュビロ磐田',
       '名古屋グランパスエイト', '京都パープルサンガ', 'ガンバ大阪', 'ヴィッセル神戸', 'サンフレッチェ広島',
       'セレッソ大阪', '大分トリニータ', 'アルビレックス新潟', '大宮アルディージャ', '川崎フロンターレ',
       'ヴァンフォーレ甲府', 'アビスパ福岡', '横浜FC', '名古屋グランパス', '京都サンガF.C.', 'モンテディオ山形',
       'ジェフユナイテッド市原・千葉', '湘南ベルマーレ', 'サガン鳥栖']
team_name_after = ['札幌', '仙台', '鹿島', '浦和', '千葉','柏', 'FC東京', '東京Ｖ', '横浜FM', '清水', '磐田',
       '名古屋', '京都', 'Ｇ大阪', '神戸', '広島','Ｃ大阪', '大分', '新潟', '大宮', '川崎Ｆ',
       '甲府', '福岡', '横浜FC', '名古屋', '京都', '山形',
       '千葉', '湘南', '鳥栖']

team_name_changer1 = dict(zip(team_name_before, team_name_after))


sal1['team'] = sal1['team'].apply(lambda x: team_name_changer1[x])
sal1['year'] = pd.to_numeric(sal1['year'])


# - Change the dtype of salary column to numeric

# change salary column to numeric type

for i, row in sal1.iterrows():
    year = row['year']
    sal = str(row['salary'])
    if year < 2011:
        if len(re.findall('\d', sal)) == 0:
            sal1.loc[sal1.index == i, 'salary'] = np.nan
            
        elif sal.endswith('億') or sal.endswith('億円'):
            new_sal = int(re.match('\d+', sal)[0]) * 10000
            sal1.loc[sal1.index == i, 'salary'] = new_sal
            
        elif '億' in sal and (str(sal).endswith('億')) == False:
            oku = int(sal.split('億')[0]) * 10000
            
            try:
                man = int(re.search('\d+',''.join(sal.split('億')[1].split('、')))[0])
                sal1.loc[sal1.index == i, 'salary'] = oku + man
            except:
                sal1.loc[sal1.index == i, 'salary'] = oku
            
            
        elif sal in ['東京ガス社員契約円','アマチュア契約円','不明円','不明']:
            sal1.loc[sal1.index == i, 'salary'] = np.nan
            
        else:
            try:
                sal1.loc[sal1.index == i, 'salary'] = sal1.loc[sal1.index == i, 'salary']                 .apply(lambda x: int(re.search('\d+',''.join(str(x).split('、')))[0]))
            except:
                print(i, sal)


# 2013~2018

sal2 =salary2.copy()

team_name_before = ['名古屋グランパス', '横浜F・マリノス', 'ガンバ大阪', 'ヴィッセル神戸', '浦和レッズ', '川崎フロンターレ',
       '柏レイソル', '鹿島アントラーズ', '大宮アルディージャ', '清水エスパルス', 'FC東京', 'サンフレッチェ広島',
       'ベガルタ仙台', 'ヴァンフォーレ甲府', 'アルビレックス新潟', 'サガン鳥栖', 'セレッソ大阪', '徳島ヴォルティス',
       'モンテディオ山形', '湘南ベルマーレ', '松本山雅FC', 'ジュビロ磐田', 'アビスパ福岡', 'コンサドーレ札幌',
       'Ｖ・ファーレン長崎']
team_name_after = ['名古屋', '横浜FM', 'Ｇ大阪', '神戸', '浦和', '川崎Ｆ','柏', '鹿島', '大宮', '清水', 'FC東京', '広島',
       '仙台', '甲府', '新潟', '鳥栖', 'Ｃ大阪', '徳島','山形', '湘南', '松本', '磐田', '福岡', '札幌','長崎']

team_name_changer2 = dict(zip(team_name_before, team_name_after))

sal2['team'] = sal2['team'].apply(lambda x: team_name_changer2[x])
sal2['year'] = pd.to_numeric(sal2['year'])

for i, row in sal2.iterrows():
    sal = row['salary']
    
    if '億' in sal:
        if '億円' in sal:
            oku = int(sal.split('億')[0]) * 10000
            sal2.loc[sal2.index == i , 'salary'] = oku
        else:
            
            oku = int(sal.split('億')[0]) * 10000
            man = int(sal.split('億')[1].split('万')[0])
            sal2.loc[sal2.index == i , 'salary'] = oku + man
    else:
        sal2.loc[sal2.index == i , 'salary'] = int(sal.split('万')[0])

sal2 = sal2[sal2['year'] > 2013].drop(['ranking'], 1)


# Merge Salary Data to Main Dataset

# - I merged match reports dataset, which is given by competition and shows the players of that match, to main dataset.
# Then, I merged salary dataset using season and players columns in the main dataset.
sal_df = pd.concat([sal1, sal2], ignore_index = True)
sal_df['salary'] = pd.to_numeric(sal_df['salary'], errors='coerce')


reports = pd.read_csv(path + 'ex_match_reports.csv')

home_df = df[['match_Year', 'home_team', 'home_team_player1', 'home_team_player2', 'home_team_player3','home_team_player4',
 'home_team_player5','home_team_player6','home_team_player7','home_team_player8','home_team_player9', 'home_team_player10','home_team_player11' ]]

away_df = df[['match_Year', 'away_team','away_team_player1', 'away_team_player2', 'away_team_player3', 'away_team_player4', 'away_team_player5', 'away_team_player6', 'away_team_player7',
 'away_team_player8','away_team_player9','away_team_player10','away_team_player11' ]]


home_players = pd.melt(frame=home_df, id_vars=['match_Year', 'home_team'], var_name = 'number', value_name = 'player').drop(['number'], 1)
home_players.loc[home_players.player.notnull(), 'player'] = home_players.loc[home_players.player.notnull(), 'player'].apply(lambda x: re.search('\D+', x)[0].strip()[:-3])

away_players = pd.melt(frame=away_df, id_vars=['match_Year', 'away_team'], var_name = 'number', value_name = 'player').drop(['number'], 1)
away_players.loc[away_players.player.notnull(), 'player'] = away_players.loc[away_players.player.notnull(), 'player'].apply(lambda x: re.search('\D+', x)[0].strip()[:-3])


player1 = home_players.drop_duplicates(subset=['match_Year', 'home_team', 'player']).rename(columns={'home_team': 'team'})
player2 = away_players.drop_duplicates(subset=['match_Year', 'away_team', 'player']).rename(columns={'away_team': 'team'})

player_df = pd.concat([player1, player2], ignore_index=True).dropna()
player_df = player_df[player_df['match_Year'] > 2001].drop_duplicates(subset=['match_Year', 'team', 'player'])


# - The format of players' name in two datasets were different, so I needed to unificate it.
# unificate format of players' name in datasets
player_df['player'] = player_df['player'].apply(lambda x: ''.join(str(x).split(' ')))
sal_df['player'] = sal_df['player'].apply(lambda x: ''.join(str(x).split('　')))

# unificate same chinese characters with different shapes in two datasets
kanji_converter = {'﨑': '崎', '眞': '真', '澤': '沢'}
for df_ in [sal_df, player_df]:
    for i, row in df_.iterrows():
        name = row['player']
        for kanji in list(kanji_converter.keys()):
            if kanji in name:
                new_name = []
                for x in list(name):
                    try:
                        y = kanji_converter[x]
                        new_name.append(y)
                    except:
                        new_name.append(x)
                new_name = ''.join(new_name)

                df_.loc[df_.index == i, 'player'] = new_name

player_sal_df = player_df.merge(sal_df, how='left', left_on=['match_Year', 'team', 'player'],
                               right_on=['year', 'team', 'player'])

player_sal_df.loc[player_sal_df.player =='フェルナンドトーレス', 'salary'] = 80000
player_sal_df.loc[player_sal_df.player =='アンドレスイニエスタ', 'salary'] = 320000

player_sal_df = player_sal_df.dropna(subset=['salary'])


# - Merging finished salary dataset to main dataset
player_cols =['home_team_player1', 'home_team_player2', 'home_team_player3','home_team_player4','home_team_player5',
                 'home_team_player6','home_team_player7','home_team_player8','home_team_player9', 'home_team_player10','home_team_player11',
                 'away_team_player1', 'away_team_player2', 'away_team_player3', 'away_team_player4', 'away_team_player5', 'away_team_player6',
                 'away_team_player7','away_team_player8','away_team_player9','away_team_player10','away_team_player11']

for col in player_cols:
    df.loc[df[col].notnull(), col] =     df.loc[df[col].notnull(), col].apply(lambda x: re.search('\D+', x)[0].strip()[:-3]) 
    df.loc[df[col].notnull(), col] = df.loc[df[col].notnull(), col].apply(lambda x: ''.join(str(x).split(' ')))


# - I needed to use this way to merge salary date, because of memory error. 
merging_df = df.copy()[['id','match_date']]

for num in range(1, 12):
    num = str(num)
    
    for place in ['home', 'away']:
        col = '{}_team_player{}'.format(place, num)
        temp = df[['id', 'match_Year', '{}_team'.format(place), col]].merge(sal_df, how='left', left_on = ['match_Year', '{}_team'.format(place), col], 
                                                            right_on = ['year', 'team', 'player'])
        gc.collect()
        temp = temp[['id', 'salary']]
        merging_df = merging_df.merge(temp, how='left', on = 'id')
        merging_df.rename(columns={'salary': '{}_team_player{}_salary'.format(place, num)}, inplace=True)
        gc.collect()
        
df = df.merge(merging_df.drop(['match_date'],1), how='left', on='id')


# Feature Engineering using Salary Features

# Average Salary of Teams per Season

# - The number of players with salary information are different between each team. If we want to calculate the average salary, we need to determine the number of players that will be used for calculation.
# - I concluded that it is good to calculate the salary of top 11 players, since they can show the level of team and reflect the popularity of team indirectly (Team with higher salary are expected to perform better.)
sal_temp = sal_df.copy()
sal_temp['salary_11'] = np.nan

teams = list(sal_temp['team'].unique())
years = range(2002, 2019)

for team_ in teams:
    for year in years:
        salary_data = sorted(sal_temp.loc[(sal_temp['team'] == team_)&(sal_temp['year'] == year), 'salary'].values.tolist())
        
        if len(salary_data) == 0:
            continue
        else:
            crt = salary_data[-11]
            temp = sal_temp[(sal_temp['team'] == team_)&(sal_temp['year']== year)]
        
        for i, row in temp.iterrows():
            sal = row['salary']
            
            
            if sal >= crt:
                sal_temp.loc[sal_temp.index == i, 'salary_11'] = 1
            else:
                sal_temp.loc[sal_temp.index == i, 'salary_11'] = 0

players_11 = sal_temp[sal_temp.salary_11 == 1].sort_values(by='year')

team_avg_sal = players_11.groupby(['year', 'team'])['salary'].agg('mean').reset_index()
team_avg_sal['year'] += 1      # To meet the competition rule


df = df.merge(team_avg_sal, how='left', left_on = ['match_Year', 'home_team'], 
                               right_on = ['year', 'team']).rename(columns={'salary': 'last_year_home_team_avg_salary'}).drop(['team', 'year'], 1)

df = df.merge(team_avg_sal, how='left', left_on = ['match_Year', 'away_team'], 
                                    right_on = ['year', 'team']).rename(columns={'salary': 'last_year_away_team_avg_salary'}).drop(['team', 'year'], 1)


# Highest Salary of Team's Starting Lineup

home_sal_cols = ['home_team_player10_salary', 'home_team_player11_salary', 
                 'home_team_player1_salary', 'home_team_player2_salary', 'home_team_player3_salary', 
                 'home_team_player4_salary', 'home_team_player5_salary', 'home_team_player6_salary', 
                 'home_team_player7_salary', 'home_team_player8_salary', 'home_team_player9_salary']

away_sal_cols = ['away_team_player10_salary', 'away_team_player11_salary', 'away_team_player1_salary', 
                 'away_team_player2_salary', 'away_team_player3_salary', 'away_team_player4_salary', 
                 'away_team_player5_salary', 'away_team_player6_salary', 'away_team_player7_salary', 'away_team_player8_salary', 
                 'away_team_player9_salary']

df['home_team_highest_salary'] = np.nan
df['away_team_highest_salary'] = np.nan

for i, row in df.iterrows():
    if row['match_Year'] < 2002 or row['division'] == 2:
        continue
    
    else:
        try:
            home_max = np.max([x for x in row[home_sal_cols].values.tolist() if ~np.isnan(x)])
            away_max = np.max([x for x in row[away_sal_cols].values.tolist() if ~np.isnan(x)])
        except:
#             print(i)
            continue
        
        df.loc[df.index == i, 'home_team_highest_salary'] = home_max
        df.loc[df.index == i, 'away_team_highest_salary'] = away_max


# Changed the 0 attendance value using lag features
year_lag_cols = ['home_avg_attendance_by_year_lag1','home_avg_attendance_by_year_lag2','home_avg_attendance_by_year_lag3','home_avg_attendance_by_year_lag4']
df.loc[df['attendance'] == 0, 'attendance'] = np.mean(df.loc[df['attendance'] == 0, year_lag_cols].values) * (df.loc[df['attendance'] == 0, 'capacity'].values)

df.to_csv(path + 'fea_eng2.py', index=False)

