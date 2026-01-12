import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import os

def display(obj):
    print(obj)

# --- Load Data or Generate Mock Data ---
data_file = "banks_reshaped.parquet"

if os.path.exists(data_file):
    print(f"Loading data from {data_file}...")
    DF = pd.read_parquet(data_file)
else:
    print(f"File {data_file} not found. Generating mock data...")
    np.random.seed(42)
    n_banks = 100
    n_years = 5
    countries = ['US', 'UK', 'DE']
    years = list(range(2018, 2018 + n_years))

    data = []
    for bank_id in range(n_banks):
        country = np.random.choice(countries)
        for year in years:
            # Base values
            nim = np.random.normal(3, 1)
            roa = np.random.normal(1, 0.5)
            roe = roa * 10 + np.random.normal(0, 2)
            eff = 60 + np.random.normal(0, 10)
            fee = np.random.uniform(10, 50)
            dep_dem = np.random.uniform(10, 40)
            dep_sav = np.random.uniform(10, 40)
            dep_time = 100 - dep_dem - dep_sav + np.random.normal(0, 5)
            branches = np.random.randint(10, 1000)
            dbranches = np.random.normal(0, 5)

            # Introduce some outliers
            if np.random.random() < 0.05:
                nim = 100 # outlier
            if np.random.random() < 0.05:
                roe = -500 # outlier

            # Introduce missing data
            if np.random.random() < 0.01:
                country = np.nan

            data.append({
                'bank_id': bank_id,
                'year': year,
                'country': country,
                'NIM': nim,
                'ROA': roa,
                'ROE': roe,
                'Efficiency_Ratio': eff,
                'Fee_Income_Share': fee,
                'Dep_Demand_Share': dep_dem,
                'Dep_Savings_Share': dep_sav,
                'Dep_Time_Share': dep_time,
                'Branches': branches,
                'dBranches_pct': dbranches
            })

    DF = pd.DataFrame(data)
    print(f"Mock data generated: {DF.shape}")

# ================== READY-TO-RUN TESTS (WITH SAFETY FILTERS) ==================

# ---------- 1) Denominator sanity screens ----------
# These thresholds are conservative; adjust if you want.
# Goal: prevent ROE/Efficiency blowing up due to equity/oprev ~ 0.
DF2 = DF.copy()

# If you have raw components in DF, use them. If not, we screen based on ratio plausibility only.
# Ratio plausibility filters (keep broad but remove pathological values):
ratio_bounds = {
    "NIM": (-20, 20),
    "ROA": (-20, 20),                 # bank ROA in %
    "ROE": (-200, 200),               # wide, but removes absurd outliers
    "Efficiency_Ratio": (0, 300),      # cost-to-income usually 20–150; keep wide
    "Fee_Income_Share": (0, 500),      # wide; remove insane
    "Dep_Demand_Share": (0, 300),
    "Dep_Savings_Share": (0, 300),
    "Dep_Time_Share": (0, 300),
}

for v, (lo, hi) in ratio_bounds.items():
    if v in DF2.columns:
        DF2.loc[~DF2[v].between(lo, hi), v] = np.nan

# Also drop bank-years with missing core identifiers
DF2 = DF2.dropna(subset=["bank_id", "year", "country"]).copy()

print("After plausibility screens (set to NaN, not dropped):")
desc_df = DF2[["NIM","ROA","ROE","Efficiency_Ratio"]].describe(percentiles=[.01,.5,.99]).T[["count","mean","50%","1%","99%"]]
print(desc_df)
desc_df.style.to_latex("descriptive_stats.tex")

# ---------- 2) Optional: re-winsorize after screens ----------
def winsorize_series(s, p_lo=0.01, p_hi=0.99):
    s = pd.to_numeric(s, errors='coerce')
    if s.isna().all():
        return s
    lo, hi = s.quantile(p_lo), s.quantile(p_hi)
    return s.clip(lo, hi)

for v in ["NIM","ROA","ROE","Efficiency_Ratio","Fee_Income_Share",
          "Dep_Demand_Share","Dep_Savings_Share","Dep_Time_Share","dBranches_pct"]:
    if v in DF2.columns and DF2[v].notna().any():
        DF2[v] = winsorize_series(DF2[v], 0.01, 0.99)

# ---------- 3) Tests ----------
# (A) Correlations (Pearson + Spearman)
test_vars = ["Branches","dBranches_pct","NIM","ROA","ROE","Efficiency_Ratio",
             "Fee_Income_Share","Dep_Demand_Share","Dep_Savings_Share","Dep_Time_Share"]
test_vars = [v for v in test_vars if v in DF2.columns and DF2[v].notna().any()]

corr_df = DF2[test_vars].copy()
pearson = corr_df.corr(method="pearson")
spearman = corr_df.corr(method="spearman")

print("\n================ CORRELATION (PEARSON) ================")
display(pearson)
pearson.style.to_latex("correlation_pearson.tex")

print("\n================ CORRELATION (SPEARMAN) ================")
display(spearman)
spearman.style.to_latex("correlation_spearman.tex")

# (B) Mean-difference tests across countries (ANOVA + Kruskal)
print("\n================ COUNTRY DIFFERENCES ================")
country_stats = []
for v in ["NIM","ROA","ROE","Efficiency_Ratio","Fee_Income_Share"]:
    if v not in DF2.columns:
        continue
    g = DF2.dropna(subset=[v]).groupby("country")[v].apply(list)
    if g.shape[0] < 2:
        continue
    # ANOVA (parametric)
    try:
        fstat, pval = stats.f_oneway(*g.values)
    except Exception:
        fstat, pval = (np.nan, np.nan)
    # Kruskal (non-parametric)
    try:
        hstat, pval_k = stats.kruskal(*g.values)
    except Exception:
        hstat, pval_k = (np.nan, np.nan)

    print(f"{v:>18s} | ANOVA p={pval:.4g} | Kruskal p={pval_k:.4g} | Ns: { {k:len(vv) for k,vv in g.items()} }")
    country_stats.append({
        "Variable": v,
        "ANOVA p-value": pval,
        "Kruskal p-value": pval_k,
    })

if country_stats:
    pd.DataFrame(country_stats).style.to_latex("country_differences.tex")

# (C) Baseline panel regression template (optional)
# Requires linearmodels; uses entity FE, year FE.
try:
    from linearmodels.panel import PanelOLS
    import statsmodels.api as sm

    reg_df = DF2.dropna(subset=["ROA","NIM","Efficiency_Ratio","Fee_Income_Share"]).copy()
    reg_df = reg_df.set_index(["bank_id","year"])

    y = reg_df["ROA"]
    X = reg_df[["NIM","Efficiency_Ratio","Fee_Income_Share"]]
    X = sm.add_constant(X)

    mod = PanelOLS(y, X, entity_effects=True, time_effects=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)

    print("\n================ BASELINE PANELOLS (ROA) ================")
    print(res.summary)

    with open("panel_regression.tex", "w") as f:
        f.write(res.summary.as_latex())

except Exception as e:
    print("\nPanel regression skipped (missing package or data issue):", str(e))
