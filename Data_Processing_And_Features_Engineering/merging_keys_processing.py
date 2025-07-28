import pandas as pd
import wrds
from data_processing.preprocessing_functions import *

# 1) Connect to WRDS and pull CCMXPF_LNKUSED
db = wrds.Connection()

lnkused = db.get_table(library='crsp', table='ccmxpf_lnkused')

#Export raw data in csv:
lnkused.to_csv("output.csv", index=False)

#Rename collumns for clarity
lnkused = pd.read_csv(
    r"C:\Etudes\EPFL\1ère\MA2\ML\PROJECT\data_raw\Merging_keys_raw.csv",
    parse_dates=["ulinkdt", "ulinkenddt"]
).rename(columns={
    "ugvkey":     "gvkey",
    "upermno":    "permno",
    "ulinkdt":    "linkdt",
    "ulinkenddt": "linkenddt",
    "ulinkprim":  "linkprim",   
    "ulinktype":  "linktype"    
})

#count unic gvkeys
total_gvkeys = lnkused["gvkey"].nunique()
print(total_gvkeys)

#how many distinct GVKEYs ever appear with a “non‐P” flag.
nonprim_gvkeys = lnkused.loc[lnkused["linkprim"].isin(["C","j","N"]), "gvkey"].unique()
print(len(nonprim_gvkeys))

#checking how many ulinktype there are
ulinktype_counts = lnkused["ulinktype"].value_counts()
print(ulinktype_counts)

#Avoiding problematic spin-offs:
linkused = linkused.query("linktype in ['LC','LU']")

#Checking wether there might be some Na in the ulinkprim category
print(lnkused["linkprim"].isna().sum())

#how many row of each ulinkprim there are:
print(lnkused["linkprim"].value_counts())

#P: Primary common share
#C: Co-primary share
#J: Link exist because of complexe corporate event
#N: Often this appears when a firm changes share codes mid‐quarter, or for a very brief test listing that never became the main security.

# 2. Convert/link‐date columns to timestamps, fill NaT end dates
lnkused["linkdt"]    = pd.to_datetime(lnkused["linkdt"])
lnkused["linkenddt"] = pd.to_datetime(lnkused["linkenddt"])

#Modifying the Nan (still active key) so that it's easier to filter when we merge the data:
lnkused["linkenddt"] = lnkused["linkenddt"].fillna(pd.Timestamp("2099-12-31"))

# 3. Keep only the “used” links :
lnkused = lnkused[lnkused["usedflag"] == 1]

#Keeping P C and J that will be One hot encoded to try to extract relevant signal
lnkused = lnkused[lnkused["linkprim"].isin(["P", "C", "J"])] 

# 4. Select just the columns we need
merging_ready = lnkused[[
    "gvkey",      # Compustat firm ID
    "permno",     # CRSP security ID
    "linkdt",     # Link start
    "linkenddt",  # Link end
    "linkprim"    #Kind of securities (P,C,J)
]]

#Export cleaned and ready merging keys
merging_ready.to_csv("link_ready.csv", index = False)
