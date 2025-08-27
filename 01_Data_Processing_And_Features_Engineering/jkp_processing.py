import pandas as pd
import numpy as np
import itertools
from data_processing.preprocessing_functions import *

data = pd.read_csv("JKP.csv", delimiter=",")
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
data

df = data.pivot(index='date', columns='name', values='ret').sort_index()
# delete dates before 1961-04-30, to match compustat data
df = df[df.index >= '1961-04-30']
df


# print the columns names with missing values and the number of missing values
missing_values = df.isnull().sum()
missing_columns = missing_values[missing_values > 0]
print("Columns with missing values and their counts:")
print(missing_columns)

# flag matrice that marks missing values with 1 and non-missing values with 0
flag_matrix = df.isna().astype(int)

detailed_blocks = []
for col in df.columns:
    blocks = nan_blocks_info(df[col])
    for block in blocks:
        detailed_blocks.append({
            'factor': col,
            'start_date': block['start_date'],
            'end_date': block['end_date'],
            'length': block['length']
        })

detailed_df = pd.DataFrame(detailed_blocks)
print(detailed_df)

# fill with 0 all blocks that start on 1961-04-30
target_start = pd.Timestamp('1961-04-30')
for factor in df.columns:
    blocks = nan_blocks_info(df[factor])
    for block in blocks:
        if block['start_date'] == target_start:
            df.loc[block['start_date']:block['end_date'], factor] = 0.0

# forward fill les NaN restants
df.fillna(method='ffill', inplace=True)


print("Nombre de NaN restant par colonne :")
df.isna().sum()

# add the flag matrix to the original dataframe
df_clean = pd.concat([df, flag_matrix.rename(columns=lambda col: f"{col}_flag")], axis=1)
df_clean.to_csv("JKP_clean.csv", index=True)
df_clean



