from data_processing.preprocessing_functions import *
import pandas as pd
import numpy as np

compustat = pd.read_csv(r"C:\Etudes\EPFL\1ère\MA2\ML\PROJECT\data_raw\Predictors\CompFirmCharac.csv\CompFirmCharac.csv")

# Count and list columns ending with 'y' (See what are the year to date collomnus among those)
cols_ending_with_y = [col for col in compustat.columns if col.endswith('y')]
count_ending_with_y = len(cols_ending_with_y)

print(f"Number of columns ending with 'y': {count_ending_with_y}")
print("Columns ending with 'y':", cols_ending_with_y)


# collumns type
mixed_df = identify_string_and_mixed_columns(compustat)
print(mixed_df.to_string(index=False))


#what are the value of fic
result = value_counts_for_column(compustat, 'fic')
result.head()


#Drop all the collumns with missing value >= 60%
compustat = drop_cols_by_missing_raiot(compustat,0.6)

#dropping the first incorporation / cik number / exchange 
compustat = compustat.drop(columns=['fic','cik','exchg'])

#hot enconding the information of the currency in which the financial data is available (may be an interesting signal), and dropping curdcq
one_hot = pd.get_dummies(compustat['curcdq'], prefix='curcdq')
one_hot = one_hot.astype(int)   # True→1, False→0
compustat = pd.concat(
    [compustat.drop(columns=['curcdq']), one_hot],
    axis=1
)

#Convert data to datetime:
compustat['datadate'] = pd.to_datetime(compustat['datadate'])


#Keep only industrial firm (avoid distortion of classic ratio and consistency with benchmark studies as JKP) (already done in the dataset)
compustat = compustat[compustat['indfmt'] == 'INDL'].drop(columns='indfmt')


# Keep only consolidated financial statement to avoid double counting (already done in the dataset)
compustat = compustat[compustat['consol'] == 'C'].drop(columns='consol')


# Keep only GAAP Standard format: Ensure consistency (already done in the dataset)
compustat = compustat[compustat['datafmt'] == 'STD'].drop(columns='datafmt')


# Drop raw identifier exept gvkey:
compustat = compustat.drop(columns=['tic','cusip','conm'])

# Keep only active firms
compustat = compustat[compustat['costat'] == 'A'].drop(columns='costat')

#check Duplicate paris
print_duplicate_pairs(compustat, 'gvkey', 'datadate')

#Drop duplicates based on gvkey and datadate, keeping the row with the largest fyearq
# 1) Sort so that, within each (gvkey, datadate), the row with the HIGHEST fyearq is first
compustat = compustat.sort_values(
    by=["gvkey", "datadate", "fyearq"],
    ascending=[True, True, False]
)

# 2) Drop duplicates on gvkey+datadate, keeping the first (i.e. largest fyearq) in each group
compustat = compustat.drop_duplicates(
    subset=["gvkey", "datadate"],
    keep="first"
).reset_index(drop=True)

#Check:
print_duplicate_pairs(compustat, 'gvkey', 'datadate')

#check:
print_constant_columns(compustat)

#drop
compustat = drop_single_value_cols(compustat)

#Heavy process: Working by chunk (from the cluster)
input_path = 'compustat_small.csv'
output_path = 'CompFirmCharac_all_with_quarters.csv'
chunk_size = 100_000
reader = pd.read_csv(input_path, chunksize=chunk_size)

#Transforming YtD collumns into quarter collumns: 
with open(output_path, 'w') as f_out:
    for i, chunk in enumerate(reader):
        print(f"Processing chunk {i}...")
        converted = convert_chunk(chunk)
        converted.to_csv(f_out, index=False, header=(i == 0))


#Drop useless collumns generate from the quarterization
compustat.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], inplace=True)

#dataqtr and datafqtr are not usefull in our case we assume the data is directly available at t + other collumns no needed
compustat = compustat.drop(columns=['datacqtr', 'datafqtr','fyr','fyearq' ])

#Check Nan:
count_nan_columns_by_threshold(compustat)
count_nan_rows_by_threshold(compustat)

#Check columns name:
print_column_name(compustat)

#earliest datadate so that we can cut jkp factors dataset:
print_earliest_date(compustat, 'datadate')

# Building a Flag Matrix before filling Nan to feed the model
#The aim is to indicate to the model that those values are to be taken with caution

flag_matrix = compustat.isna().astype(int)
#change the the second occurence of gvkey and datadate collumn title with 'gvkey_flag' and 'datadate_flag': 
flag_matrix = flag_matrix.rename(columns={'gvkey': 'gvkey_flag', 'datadate': 'datadate_flag'})

#flag matrix with compustat matrix but I want to keep as in the original datraframe the values of gvkey and datadate:
flag_matrix = pd.concat([compustat[['gvkey', 'datadate']], flag_matrix], axis=1)

#rename collumns:

# columns to leave as-is
exclude = {'gvkey', 'datadate', 'gvkey_flag', 'datadate_flag'}

# build a rename map for everything else
rename_map = {
    col: f"{col}_flag"
    for col in flag_matrix.columns
    if col not in exclude
}

# apply it
flag_matrix = flag_matrix.rename(columns=rename_map)

#Export separatly the flag matrix
flag_matrix.to_csv('compustat_flag_matrix.csv', index=False)

#Nan_filling:
compustat = fill_groupwise_all_nan(compustat, group_col='gvkey', fill_value=0)


#Filling initial block of zero and Nan bloc by 0 until a NonNan and Non 0 value encountered
# 1) Sort by gvkey, then by datadate (ascending)
compustat = compustat.sort_values(['gvkey', 'datadate'])

# 2) Group by gvkey and apply the fill function
compustat = (
    compustat
    .groupby('gvkey', group_keys=False)
    .apply(fill_initial_zeros_and_nans)
)

#Filling block of Nan from the start with 0 until not Nan or Not 0 value encoutered
fill_backwards_until_valid(compustat)

#imputing cross sectional median and 0 if the CSM is 0
impute_med_then_zero_inplace(compustat, 'datadate', 'gvkey')

#Export Data:
compustat.to_csv("compustat_quarterly_withoutNan.csv") #IF ALL GOOD DO THIS


#Agregating compustat and the flag matrix
# 1) Both with the same index
flag_matrix.index = compustat.index

# 2) Concatenating
combined = pd.concat([compustat, flag_matrix], axis=1)

# 3) Creating a new collumn order, for each original collun we insert the flag collumn next
new_order = []
for col in compustat.columns:
    new_order.append(col)
    flag_col = f"{col}_flag"
    if flag_col in combined.columns:
        new_order.append(flag_col)

# 4) Reordering
combined = combined[new_order]

# keep only the first of any identical column names
combined = combined.loc[:, ~combined.columns.duplicated()]

#Export final version with flags:
combined.to_csv('compustat_quarterly_withoutNan_withFlag.csv')