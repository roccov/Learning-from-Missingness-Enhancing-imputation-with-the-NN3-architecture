import pandas as pd
import numpy as np
from pandas import DataFrame
#---------------------------------------------------------------------------

def identify_string_and_mixed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame listing each column that contains non-numeric (object) or mixed-type values,
    along with its pandas dtype and a sample of Python types observed.
    """
    result = []
    for col in df.columns:
        if df[col].dtype == 'object':
            # collect up to 5 unique Python types in this column
            sample_types = df[col].dropna().map(type).unique()[:5].tolist()
            result.append({
                'column': col,
                'pandas_dtype': df[col].dtype,
                'sample_types': sample_types
            })
    return pd.DataFrame(result)

#----------------------------------------------------------------------

def value_counts_for_column(df: DataFrame, column: str) -> DataFrame:
    """
    Return a DataFrame listing each unique value in the specified column of df
    along with its occurrence count, sorted by count descending.

    """
    counts = df[column].value_counts(dropna=False)
    return counts.rename_axis('value').reset_index(name='count')


#-----------------------------------------------------------------------------


def drop_cols_by_missing_ratio(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Return a copy of `df` with columns removed when the fraction of missing values
    in that column is >= threshold.
    """
    # compute missing‐value ratio per column and keep those <= threshold
    keep_cols = df.isnull().mean(axis=0) < threshold
    return df.loc[:, keep_cols]

#------------------------------------------------------------------------------------
def print_duplicate_pairs(df: pd.DataFrame, col1, col2):
    """
    Groups the DataFrame by col1 and col2, finds pairs with more than one row,
    and prints them along with their counts.

    """
    # Make sure the columns exist
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError(f"One or both columns not found: {col1}, {col2}")

    duplicate_counts = (
        df
        .groupby([col1, col2])
        .size()
        .reset_index(name='count')
    )

    duplicate_pairs = duplicate_counts[duplicate_counts['count'] > 1]
    print(duplicate_pairs)

#------------------------------------------------------------------------------------------

def print_constant_columns(df:pd.DataFrame):
    """
    Print all columns of the DataFrame that contain a single unique value (including NaN).
    
    For each such column, outputs the column name and the constant value it holds.
    """
    
    const_series = df.loc[:, df.nunique(dropna=False) == 1].iloc[0]
    print(const_series.to_string())

#-------------------------------------------------------------------------------------------------


def drop_single_value_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of `df` with columns removed if they contain only a single unique value
    (NaNs count as a unique value here, so an all-NaN column will be dropped too).
    """
    # find columns with exactly one unique value (counting NaN as a value)
    to_drop = [col for col in df.columns if df[col].nunique(dropna=False) == 1]
    return df.drop(columns=to_drop)

#---------------------------------------------------------------------------------------------------

def count_high_nan_columns(df: pd.DataFrame, threshold):
    """
    Count how many columns in df have more than `threshold` fraction of missing values.
    
    """
    return (df.isna().mean() > threshold).sum()

#--------------------------------------------------------------------------------------------------


def count_nan_columns_by_threshold(df: pd.DataFrame, thresholds=None) -> dict:
    """
    Count how many columns in `df` have a proportion of NaNs strictly greater than each threshold.

    """
    if thresholds is None:
        thresholds = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50, 0.40, 0.30,0.20,0.10]
    na_frac = df.isna().mean()
    return {thr: int((na_frac >= thr).sum()) for thr in thresholds}

#------------------------------------------------------------------------------------------------------

def count_nan_rows_by_threshold(df: pd.DataFrame, thresholds=None) -> dict:
    """
    Count how many rows in `df` have a proportion of NaNs greater than or equal to each threshold.


    """
    if thresholds is None:
        thresholds = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]
    na_frac_rows = df.isna().mean(axis=1)
    return {thr: int((na_frac_rows >= thr).sum()) for thr in thresholds}

#----------------------------------------------------------------------------------------------------------

def print_column_name(df: pd.DataFrame):
    """
    Prints the total number of columns in the DataFrame and lists all column names.

    Parameters:
    - df: pandas.DataFrame whose columns you want to inspect
    """
    column_names = df.columns.tolist()
    print(f"Total columns: {len(column_names)}")
    print(column_names)

    
#-------------------------------------------------------------------------------------------------------

def print_earliest_date(df : pd.DataFrame, date_col):
    """
    Ensures the specified date column is datetime, then prints the earliest date.

    """
    if date_col not in df.columns:
        raise ValueError(f"Column not found: {date_col}")

    # Convert to datetime, coercing errors
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # Find the earliest (minimum) date
    earliest = df[date_col].min()
    if pd.isna(earliest):
        print(f"No valid dates found in column '{date_col}'.")
    else:
        print(f"Earliest {date_col}: {earliest.date()}")

#--------------------------------------------------------------------------------------------------

def print_total_nans(df :pd.DataFrame ):
    total_nans = df.isna().sum().sum()
    print(f"Total number of NaNs in the DataFrame: {total_nans}")

#---------------------------------------------------------------------------------------------------

def fill_groupwise_all_nan(
    df : pd.DataFrame,
    group_col,
    fill_value):
    """
    For each group in `df` (grouped by `group_col`), find columns that are
    entirely NaN for that group, and replace those NaNs with `fill_value`.
    Other NaNs (in columns where the group has at least one real value)
    are left untouched.
    """
    # Build a boolean DataFrame, same shape as `df`, where
    #   mask[i,j] == True  ⇔  column j in group df.loc[df[group_col]==group,i] was all NaN
    fill_mask = (
        df
        .groupby(group_col)
        .transform(lambda col: col.isna().all())
    )
    # Wherever mask is True, replace with fill_value; else keep original
    return df.mask(fill_mask, fill_value)

#------------------------------------------------------------------------------------------------------------

def fill_initial_zeros_and_nans(group):
    # Identify which columns to process (skip the keys)
    cols = [c for c in group.columns if c not in ('gvkey', 'datadate')]

    for col in cols:
        s = group[col]
        
        # 1) “valid” becomes True once we hit a non-NaN AND non-zero
        valid = s.notna() & (s != 0)
        # 2) valid.cumsum() stays 0 until the first True in `valid`
        running_valid_count = valid.cumsum()
        # 3) mask_initial is True for rows that are NaN or 0, before any valid appears
        mask_initial = (s.isna() | (s == 0)) & (running_valid_count == 0)
        # 4) wherever mask_initial is True, force a 0; otherwise keep the original
        group[col] = s.where(~mask_initial, 0)

    return group

#------------------------------------------------------------------
def fill_backwards_until_valid(compustat: pd.DataFrame) -> None:
    """
    In-place : pour chaque gvkey et chaque colonne (hors 'gvkey','datadate'),
    remplace tous les NaN ou 0 en partant de la date la plus récente et
    en remontant dans le temps, jusqu’à la première valeur non-NaN et non-0.
    Après cette première valeur valide, tout reste inchangé.
    """
    # 1) Computing order of the index
    idx_desc = (
        compustat
        .sort_values(['gvkey','datadate'], ascending=[True, False])
        .index
    )
    # 2) Columns we want to treaat
    cols = [c for c in compustat.columns if c not in ('gvkey','datadate')]

    for col in cols:
        s = compustat[col]
        valid = s.notna() & (s != 0)
        cum_valid = (
            valid
            .loc[idx_desc]
            .groupby(compustat['gvkey'].loc[idx_desc], group_keys=False)
            .cumsum()
            .reindex(s.index)
        )
        mask = (s.isna() | (s == 0)) & (cum_valid == 0)

        compustat.loc[mask, col] = 0

#----------------------------------------------------------------------------------------------------
def impute_med_then_zero_inplace(
    df: pd.DataFrame,
    date_col: str = 'datadate',
    exclude_cols: tuple = ('gvkey',)
) -> None:
    """
    In-place: for every numeric column in df except those in exclude_cols and date_col,
    fill missing values by:
      1) cross-sectional median per date_col
      2) where that median is NaN (i.e. all firms missing), fill with 0
    """
    # 1) pick out the numeric columns to touch
    num_cols = (
        df
        .select_dtypes(include='number')
        .columns
        .difference(exclude_cols + (date_col,))
    )

    # 2) compute the date-by-date median and align it
    cs_median = df.groupby(date_col)[num_cols].transform('median')

    # 3) first fillna with the median, then any remaining NaN with 0
    df[num_cols] = df[num_cols].fillna(cs_median).fillna(0)

#----------------------------------------------------------------------------------------------------------------

def compute_quarters_for_group(sub: pd.DataFrame, ytd_cols: list) -> pd.DataFrame:
    """
    From year to date columns to quarterly
    """
    series_list = []

    for ycol in ytd_cols:
        y = sub[ycol].astype(float).values
        q = np.full_like(y, np.nan, dtype=float)
        last_ytd = None

        for i in range(len(y)):
            if np.isnan(y[i]):
                q[i] = np.nan
            else:
                q[i] = y[i] if last_ytd is None else y[i] - last_ytd
                last_ytd = y[i]

        base = ycol[:-1]
        series_list.append(pd.Series(q, index=sub.index, name=base + '_q'))

    return pd.concat(series_list, axis=1)

#------------------------------------------------------------------------------------------------------------


def convert_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Convert data in chunk to ease the computation of the year to date -> quarter transformation
    """

    required = {'gvkey', 'fyearq', 'fqtr'}
    if missing := (required - set(chunk.columns)):
        raise KeyError(f"Missing columns: {missing}")

    id_cols = {'gvkey', 'fyearq', 'fqtr'}
    ytd_cols = [c for c in chunk.columns if c.endswith('y') and c not in id_cols]
    if not ytd_cols:
        return chunk

    chunk.sort_values(by=['gvkey', 'fyearq', 'fqtr'], inplace=True)
    
    # Ajouter les colonnes de sortie à l’avance
    for ycol in ytd_cols:
        chunk[ycol[:-1] + '_q'] = np.nan

    for (g, fy), idx in chunk.groupby(['gvkey', 'fyearq']).groups.items():
        group = chunk.loc[idx]
        flows = compute_quarters_for_group(group, ytd_cols)
        chunk.loc[idx, flows.columns] = flows

    return chunk

#--------------------------------------------------------------------------------------------------------------------------

def add_msrp_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    compute and append:
      - ret_lag1
      - mom_3, mom_6, mom_12
      - vol_12
      - downside_dev_12
      - skew_12
      - kurt_12
      - max_dd_12
      - sharpe_12
    """
    df = df.copy()

    # 1) parse & sort
    df['MthCalDt'] = pd.to_datetime(df['MthCalDt'])
    df = (
        df
        .sort_values(['PERMNO', 'MthCalDt'])
        .reset_index(drop=True)
    )

    grp = df.groupby('PERMNO')['Mth_Ex_ret']

    # 2) one‐month lag, fill first with zero
    df['ret_lag1'] = grp.shift(1).fillna(0)

    # 3) momentum
    for w in (3, 6, 12):
        df[f'mom_{w}'] = grp.transform(
            lambda x: x.rolling(window=w, min_periods=1)
                          .apply(lambda r: np.prod(1 + r) - 1, raw=True)
        )

    # 4) volatility & downside deviation
    df['vol_12'] = grp.transform(lambda x: x.rolling(12, min_periods=2).std())
    df['downside_dev_12'] = grp.transform(
        lambda x: x.rolling(12, min_periods=1)
                      .apply(lambda r: np.sqrt(np.mean(np.minimum(r, 0)**2)), raw=True)
    )

    # 5) skew & kurtosis
    df['skew_12'] = grp.transform(lambda x: x.rolling(12, min_periods=3).skew())
    df['kurt_12'] = grp.transform(lambda x: x.rolling(12, min_periods=4).kurt())

    # 6) max drawdown
    def _max_dd(r):
        cum = (1 + r).cumprod()
        return ((cum - cum.cummax()) / cum.cummax()).min()

    df['max_dd_12'] = grp.transform(
        lambda x: x.rolling(12, min_periods=1).apply(_max_dd, raw=False)
    )

    # 7) rolling momentum per volatility (Sharpe‐ratio)
    df['sharpe_12'] = df['mom_12'] / df['vol_12']

    return df

#--------------------------------------------------------------------------------------------------
def merge_crsp_with_compustat(comp: pd.DataFrame, link: pd.DataFrame, crsp: pd.DataFrame) ->pd.DataFrame :
    """
    Merges Compustat data with CRSP using the CRSP-Compustat link table and date matching 

    """

    # Merge with link table
    comp_linked = comp.merge(link, on="gvkey", how="left")

    # Filter by valid date window
    comp_linked = comp_linked[
        (comp_linked["datadate"] >= comp_linked["linkdt"]) &
        (comp_linked["datadate"] <= comp_linked["linkenddt"])
    ].reset_index(drop=True)

    # Generate date matches
    comp_linked["match1"] = comp_linked["datadate"]
    comp_linked["match2"] = comp_linked["datadate"] + pd.DateOffset(months=1)
    comp_linked["match3"] = comp_linked["datadate"] + pd.DateOffset(months=2)

    # Create expanded dataset
    df1 = comp_linked.drop(columns=["match2", "match3"]).rename(columns={"match1": "date"})
    df2 = comp_linked.drop(columns=["match1", "match3"]).rename(columns={"match2": "date"})
    df3 = comp_linked.drop(columns=["match1", "match2"]).rename(columns={"match3": "date"})
    expanded = pd.concat([df1, df2, df3], ignore_index=True)

    # Align dates to month-end
    expanded["date"] = pd.to_datetime(expanded["date"]).dt.to_period("M").dt.to_timestamp("M")
    crsp["date"] = pd.to_datetime(crsp["date"]).dt.to_period("M").dt.to_timestamp("M")

    # Merge CRSP with expanded Compustat
    merged = pd.merge(crsp, expanded, on=["permno", "date"], how="inner")

    return merged


#---------------------------------------------------------------------------
# fonction to identify blocks of NaN values 
def nan_blocks_info(series):
    """
    Returns a list of dictionaries for the NaN blocks in the Series,
    with 'start_date', 'end_date' and 'length' for each block.
    """
    is_na = series.isna().tolist()
    dates = series.index.tolist()
    blocks = []
    
    i = 0
    n = len(series)
    while i < n:
        if is_na[i]:
            start_idx = i
            # Avancer jusqu'à la fin du bloc NaN
            while i < n and is_na[i]:
                i += 1
            end_idx = i - 1
            # Conserver le bloc
            blocks.append({
                'start_date': dates[start_idx],
                'end_date': dates[end_idx],
                'length': end_idx - start_idx + 1
            })
        else:
            i += 1
            
    return blocks