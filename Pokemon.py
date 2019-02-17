import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 

# read data
pokemon_df = pd.read_csv('/Users/yueyue/Desktop/usc/M2S2/EE599/project/pokemon.csv')
combats_df = pd.read_csv('/Users/yueyue/Desktop/usc/M2S2/EE599/project/pokemon2.csv')

# split abilities using ","
pokemon_df2 = pokemon_df['abilities'].str.split(', ',expand=True)
pokemon_df=pokemon_df.drop('abilities',axis=1).join(pokemon_df2)
pokemon_df.rename(columns={0:'ability1',1:'ability2',2:'ability3',3:'ability4',4:'ability5',5:'ability6'},inplace=True)

# delete "[" and "]" in ability1 - ability6
pokemon_df['ability1']=pokemon_df['ability1'].str.replace('[','')
for i in range (6):
    pokemon_df['ability'+str(i+1)]=pokemon_df['ability'+str(i+1)].str.replace(']','')

# fill Null data in type2 with None
pokemon_df['type2'] = pokemon_df['type2'].fillna('None')
for i in range (2,7):
    pokemon_df['ability'+str(i)] = pokemon_df['ability'+str(i)].fillna('None')

# replace the only one '30 (Meteorite)255 (Core)' in capture_rate with 30
pokemon_df.loc[pokemon_df['capture_rate'] == '30 (Meteorite)255 (Core)'] = 30
pokemon_df['capture_rate']=pokemon_df['capture_rate'].astype(str).astype(int)

# encoding abilities (takse long time and has end sign)
for index, row in pokemon_df.iterrows():# establish column for each ability
    for i in range(1,7):
        if row['ability'+str(i)] not in pokemon_df.index:
            pokemon_df[row['ability'+str(i)]] = 0
for column in pokemon_df.columns:#establish abilities statistics
    for row_index,row in pokemon_df.iterrows():
        if column==row['ability1'] or column==row['ability2'] or column==row['ability3'] or column==row['ability4'] or column==row['ability5'] or column==row['ability6']:
            pokemon_df[column][row_index]+=1
print("end")

# encoding types (takse long time and has end sign)
for index, row in pokemon_df.iterrows():# establish column for each type
    for i in range(1,3):
        if row['type'+str(i)] not in pokemon_df.index:
            pokemon_df[row['type'+str(i)]] = 0
for column in pokemon_df.columns: #establish types statistics
    for row_index,row in pokemon_df.iterrows(): 
        if column==row['type1'] or column==row['type2']:
            pokemon_df[column][row_index]+=1
print("end")

# encoding classification (takse long time and has end sign)
for index, row in pokemon_df.iterrows():# establish column for each class
    if row['classfication'] not in pokemon_df.index:
        pokemon_df[row['classfication']] = 0
for column in pokemon_df.columns:  #establish classes statistics
    for row_index,row in pokemon_df.iterrows(): 
        if column==row['classfication']:
            pokemon_df[column][row_index]+=1
print("end")

# delete non-useful features
for i in range (1,7):
    pokemon_df=pokemon_df.drop('ability'+str(i),axis=1)
for i in range (1,3):
    pokemon_df=pokemon_df.drop('type'+str(i),axis=1)
pokemon_df=pokemon_df.drop('classfication',axis=1)
pokemon_df=pokemon_df.drop('None',axis=1)
pokemon_df=pokemon_df.drop('name',axis=1)
pokemon_df=pokemon_df.drop('japanese_name',axis=1)

# save to file
pokemon_df.to_csv('/Users/yueyue/Desktop/usc/M2S2/EE599/project/pokemon_new.csv')
