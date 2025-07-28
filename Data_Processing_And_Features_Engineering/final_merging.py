import pandas as pd
from data_processing.preprocessing_functions import *

#Downloads our datasets:

#Merging keys:
link = pd.read_csv(r"C:\Etudes\EPFL\1ère\MA2\ML\PROJECT\data_processed\link_ready.csv",
    parse_dates=["linkdt", "linkenddt"])

#compustat:
comp = pd.read_csv(r"C:\Etudes\EPFL\1ère\MA2\ML\PROJECT\notebooks\compustat_quarterly_withoutNan_withFlag.csv", parse_dates=["datadate"], index_col=0)

#crsp:
crsp = pd.read_csv(r"C:\Etudes\EPFL\1ère\MA2\ML\PROJECT\notebooks\crsp_post_featureENGINEERING_NoNan_with_FlagMatrix.csv", parse_dates=['MthCalDt'], index_col = 0)

# Rename columns for clarity
crsp = crsp.rename(columns={
    "PERMNO": "permno",
    "PERMNO_flag": "permno_flag",
    'MthCalDt_flag': "date_flag",
    "MthCalDt": "date",
    "MthRet":  "ret",
    "MthRet_flag": "ret_flag",
    "Mth_Ex_ret" : "Excess_ret",
    'Mth_Ex_ret_flag': "Excess_ret_flag",
    'sprtrn': "Sp_ret",
    'sprtrn_flag' : "Sp_ret_flag"
    
})

#Merging of compustat and CRSP:
merged = merge_crsp_with_compustat(comp, link, crsp)

# 8. One-hot encode “linkprim”:
merged["is_primary"  ] = (merged["linkprim"] == "P").astype(int)
merged["is_secondary"] = (merged["linkprim"] == "C").astype(int)

#drop collumns
merged = merged.drop(columns = ["permno_flag", "date_flag","Excess_ret_flag","linkenddt", "linkprim","linkdt","gvkey", "gvkey_flag", "datadate","datadate_flag","fqtr","fqtr_flag","exchg","exchg_flag"])

#renane
merged.rename(columns={"Excess_ret": "Excess_ret_target"}, inplace=True)

#Change collumn order
series = merged.pop('Excess_ret_target')
date_idx = merged.columns.get_loc('date')
merged.insert(date_idx + 1, 'Excess_ret_target', series)

#export
merged.to_csv("MERGED_COMPUSTAT_CRSP_FLAG.csv",index=0)

#JKP factors:
JKP = pd.read_csv(r"C:\Etudes\EPFL\1ère\MA2\ML\PROJECT\data_processed\JKP_clean_with_flag.csv")

#Ensure merged has a datetime‐typed date column:
merged["date"] = pd.to_datetime(merged["date"])

# 2. Ensure jkp has a datetime date:
JKP["date"] = pd.to_datetime(JKP["date"])

# Verify JKP has exactly one row per month:
assert JKP["date"].duplicated().sum() == 0, "JKP contains duplicate month‐end dates!"

# 4. Merge JKP onto merged by “date”:
merged= pd.merge(
    merged,    
    JKP,       
    on="date",
    how="left" 
)

#rename the collumn
merged.rename(
    columns={'Excess_ret_target': 'Excess_ret_feature'},
    inplace=True
)

#sort by permno and date
merged.sort_values(['permno','date'], inplace=True)

#Shift the target for predictions
merged['Excess_ret_target'] = (
    merged
      .groupby('permno')['Excess_ret_feature']   # <- single bracket
      .shift(-1)
)


#drop the Nan target given the shift
merged.dropna(subset=['Excess_ret_target'], inplace=True)

#flag for ret_t
merged["Excess_ret_feature_flag"] = 0


#Change the column order in the DF
series = merged.pop('Excess_ret_target')
date_idx = merged.columns.get_loc('date')
merged.insert(date_idx + 1, 'Excess_ret_target', series)

#export
merged.to_csv("crsp_compustat_jkp_ExRet_withFlag_noWins_noStd.csv")