import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import glob
from pathlib import Path

DATA_DIR   = Path(".")
PRED_GLOB  = "predictions_25_alpha_*.csv"
MASTER_CSV = "crsp_compustat_jkp_ExRet_withFlag_noWins_noStd.csv"

DATE_COL = "date"
PRED_COL = "y_pred"
RET_COL  = "y_true"
SP_RET   = "Sp_ret"

TOP_Q = 0.99
BOT_Q = 0.01

COST_RATE = 0.001  # 10 bps


# Fonctions 
def monthly_spread_net_returns(preds: pd.DataFrame,
                               cost_rate: float = COST_RATE) -> pd.Series:
    """Rendements mensuels long–short moins les coûts de transaction."""
    preds = preds.copy()
    preds[DATE_COL] = pd.to_datetime(preds[DATE_COL])
    preds["month"] = preds[DATE_COL].dt.to_period("M")

    prev_w = pd.Series(dtype=float)  
    net_rets = []
    months = []

    for month, grp in preds.groupby("month"):
        if len(grp) < 10:
            continue

        # classement percentiles
        grp["pct_rank"] = grp[PRED_COL].rank(pct=True, method="first")
        long_mask  = grp["pct_rank"] >= TOP_Q
        short_mask = grp["pct_rank"] <= BOT_Q

        # rendement brut du spread
        long_ret  = grp.loc[long_mask, RET_COL].mean()
        short_ret = grp.loc[short_mask, RET_COL].mean()
        spread_brut = 0.5 * long_ret - 0.5 * short_ret

        # construction des poids courants (long +0.5, short -0.5)
        n_long  = long_mask.sum()
        n_short = short_mask.sum()
        w = pd.Series(0.0, index=grp.index)
        w.loc[long_mask]  =  0.5 / n_long
        w.loc[short_mask] = -0.5 / n_short

        # turnover = somme des |w_t – w_{t-1}|
        all_idx = w.index.union(prev_w.index)
        w_curr = w.reindex(all_idx, fill_value=0.0)
        w_prev = prev_w.reindex(all_idx, fill_value=0.0)
        turnover = (w_curr - w_prev).abs().sum()

        # coût et rendement net
        cost       = turnover * cost_rate
        spread_net = spread_brut - cost

        net_rets.append(spread_net)
        months.append(month.to_timestamp())

        # mise à jour pour le mois suivant
        prev_w = w.copy()

    return pd.Series(net_rets, index=months)


def market_cumulative_return(master_csv: Path,
                             start_date="1986-07-31",
                             monthly_cost=0.0003 / 12) -> pd.Series:
    """Calcule le cumulative return du marché avec coût mensuel (e.g., TER ETF)."""
    df = pd.read_csv(master_csv, parse_dates=[DATE_COL])
    df = df.sort_values(DATE_COL)

    # filtrer à partir de start_date
    df = df[df[DATE_COL] >= pd.to_datetime(start_date)]

    # un seul Sp_ret par date
    mkt = df[[DATE_COL, SP_RET]].drop_duplicates(DATE_COL).set_index(DATE_COL)[SP_RET]

    # retirer le coût mensuel de l'ETF (TER annualisé)
    mkt_net = mkt - monthly_cost

    # cumulative return net
    cumret = (1 + mkt_net.sort_index()).cumprod() - 1
    return cumret


# Plot
plt.figure(figsize=(10, 6))

for file in sorted(glob.glob(PRED_GLOB)):
    m = re.search(r"predictions_25_alpha_([0-9.]+)\.csv$", Path(file).name)
    if not m:
        continue
    alpha = float(m.group(1))
    preds = pd.read_csv(file)

    # calcul des rendements nets mensuels
    spread_net = monthly_spread_net_returns(preds, cost_rate=COST_RATE)
    # valeur du portefeuille (normalisée pour démarrer à 1 $)
    cum_spread_net = (1 + spread_net).cumprod()
    cum_spread_net = cum_spread_net / cum_spread_net.iloc[0]

    plt.plot(cum_spread_net.index,
             cum_spread_net.values,
             label=f"α={alpha}")

# trace de la courbe marché (normalisée pour démarrer à 1 $)
cum_market = market_cumulative_return(DATA_DIR / MASTER_CSV,
                                      start_date="1986-07-31",
                                      monthly_cost=0.0003 / 12) + 1
cum_market = cum_market / cum_market.iloc[0]
plt.plot(cum_market.index,
         cum_market.values,
         color="black", linewidth=2,
         label="Market (Sp_ret)")

plt.title("Evolution of net-of-cost portfolio value: Long–Short (1%) vs. Market")
plt.xlabel("Date")
plt.ylabel("Portfolio value (starting at $1)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)

# tracer la ligne de référence à 1$
plt.axhline(1.0, color="gray", linestyle="--", linewidth=1)


plt.tight_layout()
plt.show()


# --------------------------------------------------------
pattern = re.compile(r"results_25_alpha_([0-9.]+)\.csv$")
files = sorted(p for p in DATA_DIR.glob("results_25_alpha_*.csv") if pattern.match(p.name))

frames = []
for f in files:
    alpha_str = pattern.match(f.name).group(1)
    alpha_val = float(alpha_str)
    df = pd.read_csv(f, parse_dates=["test_end"])
    df["alpha"] = alpha_val
    frames.append(df[["test_end", "r2_test", "alpha"]])

results = pd.concat(frames, ignore_index=True)

# 4. Graphe 1 : R^2 evolution per alpha 
fig, ax = plt.subplots(figsize=(10, 5))

for alpha, grp in results.groupby("alpha"):
    grp = grp.sort_values("test_end")
    ax.plot(grp["test_end"], grp["r2_test"],
            marker="o", linewidth=1, label=f"{alpha:g}")

ax.set_title("Evolution of test R² by alpha")
ax.set_xlabel("End of test period")
ax.set_ylabel("R² (test)")
ax.legend(title="alpha", ncol=2, frameon=False)
ax.grid(True, linestyle="--", alpha=0.4)
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# 5. Graphe 2 : r^2 median by alpha
median_df = (results
             .groupby("alpha")["r2_test"]
             .median()
             .reset_index()
             .sort_values("alpha"))

fig, ax = plt.subplots(figsize=(8, 4))

# tracé ligne + marqueur
ax.plot(median_df["alpha"],          # abscisses
        median_df["r2_test"],        # ordonnées
        marker="o",                  # points visibles
        linestyle="-", linewidth=1)  # ligne qui relie les points

ax.set_title("Median test R² by alpha")
ax.set_xlabel("alpha")
ax.set_ylabel("Median R² (test)")
ax.set_xticks(median_df["alpha"])
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# parameters 
DATE_COL  = "date"         # colonne date dans predictions_alpha_*.csv
PRED_COL  = "y_pred"       # prédictions
RET_COL   = "y_true"       # rendement réalisé (excess return)
TOP_Q     = 0.99           # top 5 %
BOT_Q     = 0.01           # bottom 5 %
COST_RATE = 0.001          # coût de transaction par turnover (10 bps)

# stats function 
def compute_stats(returns: pd.Series) -> pd.Series:
    """Calcule les stats principales sur une série de rendements mensuels."""
    mean_m   = returns.mean()
    std_m    = returns.std(ddof=1)
    ann_ret  = mean_m * 12
    ann_vol  = std_m  * np.sqrt(12)
    sharpe   = ann_ret / ann_vol
    t_stat   = mean_m / (std_m / np.sqrt(len(returns)))
    wealth      = (1 + returns).cumprod()
    running_max = wealth.cummax()
    drawdowns   = (wealth - running_max) / running_max
    max_dd      = drawdowns.min()

    return pd.Series({
        "mean_monthly"       : mean_m,
        "annualized_return"  : ann_ret,
        "annualized_vol"     : ann_vol,
        "sharpe"             : sharpe,
        "t_stat"             : t_stat,
        "max_drawdown"       : max_dd
    })


files   = sorted(DATA_DIR.glob("predictions_25_alpha_*.csv"))
regex   = re.compile(r"predictions_25_alpha_([0-9.]+)\.csv$")
results = []

for f in files:
    m = regex.match(f.name)
    if not m:
        continue
    alpha_val = float(m.group(1))

    df = pd.read_csv(f, parse_dates=[DATE_COL])
    if df.empty:
        continue

    # Regroupement mensuel
    df["month"] = df[DATE_COL].dt.to_period("M")
    prev_w = pd.Series(dtype=float)
    ls_monthly = []

    for _, grp in df.groupby("month"):
        if len(grp) < 10:
            continue

        grp = grp.copy()
        grp["pct_rank"] = grp[PRED_COL].rank(pct=True, method="first")
        long  = grp[grp["pct_rank"] >= TOP_Q]
        short = grp[grp["pct_rank"] <= BOT_Q]
        if long.empty or short.empty:
            continue

        # rendement brut du spread
        long_ret  = long[RET_COL].mean()
        short_ret = short[RET_COL].mean()
        spread_brut = 0.5 * long_ret - 0.5 * short_ret

        # construction des poids pour turnover
        n_long  = len(long)
        n_short = len(short)
        w = pd.Series(0.0, index=grp.index)
        w.loc[long.index]  =  0.5 / n_long
        w.loc[short.index] = -0.5 / n_short

        # turnover = somme des |w_t – w_{t-1}|
        all_idx  = w.index.union(prev_w.index)
        w_curr   = w.reindex(all_idx, fill_value=0.0)
        w_prev   = prev_w.reindex(all_idx, fill_value=0.0)
        turnover = (w_curr - w_prev).abs().sum()

        # coût et rendement net
        cost       = turnover * COST_RATE
        spread_net = spread_brut - cost

        ls_monthly.append(spread_net)
        prev_w = w.copy()

    if not ls_monthly:
        continue

    stats = compute_stats(pd.Series(ls_monthly))
    stats["alpha"] = alpha_val
    results.append(stats)

# 4. table
stats_df = (pd
            .DataFrame(results)
            .set_index("alpha")
            .sort_index()
            .round(4))

print("Statistiques long–short (top 1 % – bottom 1 %) par alpha, net de coûts :\n")
print(stats_df)