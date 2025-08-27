import pandas as pd
from data_processing.preprocessing_functions import *

monthly_crsp = pd.read_csv(r"C:\Etudes\EPFL\1ère\MA2\ML\PROJECT\data_raw\monthly_crsp.csv")

rf = pd.read_csv(r"C:\Etudes\EPFL\1ère\MA2\ML\PROJECT\notebooks\Rf.csv")

#what's the oldest date in the dataset MthCalDt?    
oldest_date = monthly_crsp['MthCalDt'].min()
print(f"The oldest date in the dataset is: {oldest_date}")

#check if there are some nan and how many per collumns:
monthly_crsp.isna().sum()

#drop the line where MthRet is Nan, we need our target
monthly_crsp = monthly_crsp.dropna(subset=['MthRet'])  

#drop useless collumns
monthly_crsp = monthly_crsp.drop(columns=['HdrCUSIP', 'CUSIP', 'TradingSymbol', 'NAICS', 'Ticker','PERMCO', 'SICCD'])

#Drop all the observation before 1961-04-30 to match earlyest compustat date
monthly_crsp = monthly_crsp[monthly_crsp['MthCalDt'] >= '1961-04-30']
#Check the number of observations after the drop:
print(monthly_crsp.shape)

#drop the columns with only one observation of MthRet: (to make predictions)
monthly_crsp = monthly_crsp.groupby('PERMNO').filter(lambda x: len(x) > 1)

print(rf.dtypes)

#Preparation to merge FAMA FRENCH RF and merge
rf['YM']            = pd.to_datetime(rf['Date'], format='%Y-%m').dt.to_period('M')
monthly_crsp['YM']  = pd.to_datetime(monthly_crsp['MthCalDt']).dt.to_period('M')
monthly_crsp = monthly_crsp.merge(rf[['YM','RF']], on='YM', how='left')
monthly_crsp = monthly_crsp.drop(columns=['YM'])

#computation of excess returns
monthly_crsp["Mth_Ex_ret"] = monthly_crsp["MthRet"] - monthly_crsp["RF"]

#Feature engineering:
monthly_crsp = add_msrp_features(monthly_crsp)

#inspect
print(monthly_crsp.loc[:, [
  'PERMNO','MthCalDt','ret_lag1',
  'mom_3','mom_6','mom_12',
  'vol_12','downside_dev_12',
  'skew_12','kurt_12',
  'max_dd_12','sharpe_12'
]].head(15))

#Export
monthly_crsp.to_csv("CRSP_post_feature_engineering.csv", index=False)

# flag matrix that give missing value 1 and no missing 0 
flag_matrix = monthly_crsp.isna().astype(int)

#change the the second occurence of gvkey and datadate collumn title with 'gvkey_flag' and 'datadate_flag': 
flag_matrix = flag_matrix.rename(columns={'PERMNO': 'PERMNO_flag', 'MthCalDt': 'MthCalDt_flag'})

#flag matrix but I want to keep as in the original datraframe the values of gvkey and datadate:
flag_matrix = pd.concat([monthly_crsp[['PERMNO', 'MthCalDt']], flag_matrix], axis=1)

#Fill the Nan due to features engineering computation ( to have meaningfull indicator with cross sectional mean)
monthly_crsp = monthly_crsp.fillna(monthly_crsp.median(numeric_only=True))

#number of nan er collumns:
print(monthly_crsp.isna().sum())

#export:
monthly_crsp.to_csv("crsp_post_featureENGINEERING_NoNan.csv")

#rename collumns:

# columns to leave as-is
exclude = {'PERMNO', 'PERMNO_flag', 'MthCalDt', 'MthCalDt_flag'}

# build a rename map for everything else
rename_map = {
    col: f"{col}_flag"
    for col in flag_matrix.columns
    if col not in exclude
}

# apply it
flag_matrix = flag_matrix.rename(columns=rename_map)

#Agregating CRSP with the flag matrix

# 1) OSame index
flag_matrix.index = monthly_crsp.index

# 2) Concatenation
combined = pd.concat([monthly_crsp, flag_matrix], axis=1)

# 3) New collumn order: 
new_order = []
for col in monthly_crsp.columns:
    new_order.append(col)
    flag_col = f"{col}_flag"
    if flag_col in combined.columns:
        new_order.append(flag_col)

# 4) use new order
combined = combined[new_order]

# keep only the first of any identical column names
combined = combined.loc[:, ~combined.columns.duplicated()]

#Export the flag
flag_matrix.to_csv("crsp_flagMatrix.csv")

#Exporte final cimbined dataset:
combined.to_csv('crsp_post_featureENGINEERING_NoNan_with_FlagMatrix.csv')