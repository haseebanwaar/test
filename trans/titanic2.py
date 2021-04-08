import pandas as pd
import numpy as np
#%%


df = pd.read_csv('models/trans/olympics.csv',index_col  = 0, skiprows=1)
for col in df.columns:
    if col[:2] == '01':
        df.rename(columns = {col : 'Gold' + col[4:]}, inplace = True)
    if col[:2] == '02':
        df.rename(columns = {col : 'Silver' + col[4:]}, inplace = True)
    if col[:2] == '03':
        df.rename(columns = {col : 'Bronze' + col[4:]}, inplace = True)
    if col[:1] == '№':
        df.rename(columns = {col : '#' + col[1:]}, inplace = True)

names_ids = df.index.str.split('\s\(')
df.index = names_ids.str[0]
df = df.drop('Totals')


df#%%
#Q1
df.iloc[0]

#%%
#Q2
df.iloc[df['Gold'].argmax()].name


#%%
#Q3
df[['Gold.1','Gold']].diff(axis =1).idxmax()[1]

#%%
#Q4
#Only include countries that have won at least 1 gold in both summer and winter.
countries_of_interest = df[(df['Gold.1']>0) & (df['Gold']>0)]

ratio = countries_of_interest[['Gold.1','Gold']].diff(axis =1)['Gold']/countries_of_interest['Gold.2']
ratio.idxmax()

#%%

def weighted_points(df):
    gold_weight = 3
    silver_weight = 3
    bronze_weight = 3
    Points = df['Gold.2']*gold_weight+df['Silver.2']*silver_weight+df['Bronze.2']*bronze_weight
    return Points

df.loc[df['Gold'].argmax()].name

