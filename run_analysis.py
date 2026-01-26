import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
import os
import warnings

# Shim for display
def display(df):
    print(df)

# Data Loading / Mocking
def get_data():
    if os.path.exists("banks_reshaped.parquet"):
        print("Loading data from banks_reshaped.parquet...")
        return pd.read_parquet("banks_reshaped.parquet")
    else:
        print("Dataset not found. Generating mock data...")
        np.random.seed(42)
        n_banks = 200
        n_years = 6
        years = np.arange(2015, 2015 + n_years)
        bank_ids = np.arange(n_banks)

        data = []
        countries = ['US', 'UK', 'DE', 'JP', 'CN']

        for bank_id in bank_ids:
            country = np.random.choice(countries)
            # Base values for the bank to ensure entity effects
            # These are raw values, so scaling them up so they can be divided later
            base_nim = np.random.uniform(1, 5) * 1e6
            base_roa = np.random.uniform(0.5, 2) * 1e6
            base_roe = np.random.uniform(5, 15) * 1e6
            base_eff = np.random.uniform(40, 70)

            branches = np.random.randint(50, 500)

            for year in years:
                # Add random noise and time trend
                nim = base_nim + np.random.normal(0, 0.5) * 1e6
                roa = base_roa + np.random.normal(0, 0.2) * 1e6
                roe = base_roe + np.random.normal(0, 2) * 1e6

                # Efficiency components
                op_rev = np.random.uniform(100, 500)
                # Ensure op_exp aligns with efficiency ratio
                op_exp = op_rev * (base_eff / 100) + np.random.normal(0, 5)

                branches = int(branches * np.random.uniform(0.95, 1.05))

                row = {
                    'bank_id': bank_id,
                    'year': year,
                    'country': country,
                    'Net_Int_margin': nim,
                    'Return_on_Avg_assets_': roa,
                    'Return_on_Avg_equity_': roe,
                    'Ttl_Oper__Exps': op_exp,
                    'Oper__revenues': op_rev,
                    'Branches': branches,
                    'Fee_Income_Share': np.random.uniform(10, 40),
                    'Dep_Demand_Share': np.random.uniform(10, 30),
                    'Dep_Savings_Share': np.random.uniform(20, 50),
                    'Dep_Time_Share': np.random.uniform(10, 40)
                }
                data.append(row)

        return pd.DataFrame(data)

DF = get_data()

# Data Preprocessing
print("\nProcessing Data...")

# Mapping and Scaling
if 'Net_Int_margin' in DF.columns:
    DF['NIM'] = DF['Net_Int_margin'] / 1e6
if 'Return_on_Avg_assets_' in DF.columns:
    DF['ROA'] = DF['Return_on_Avg_assets_'] / 1e6
if 'Return_on_Avg_equity_' in DF.columns:
    DF['ROE'] = DF['Return_on_Avg_equity_'] / 1e6

# Efficiency Ratio
if 'Ttl_Oper__Exps' in DF.columns and 'Oper__revenues' in DF.columns:
    # Check if Efficiency_Ratio already exists (from mock or parquet)
    if 'Efficiency_Ratio' not in DF.columns:
        DF['Efficiency_Ratio'] = (DF['Ttl_Oper__Exps'] / DF['Oper__revenues']) * 100

# dBranches_pct
if 'Branches' in DF.columns:
    DF = DF.sort_values(['bank_id', 'year'])
    DF['dBranches_pct'] = DF.groupby('bank_id')['Branches'].pct_change() * 100

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
# Use describe with percentiles
desc = DF2[["NIM","ROA","ROE","Efficiency_Ratio"]].describe(percentiles=[.01,.5,.99]).T[["count","mean","50%","1%","99%"]]
print(desc)

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

print("\n================ CORRELATION (SPEARMAN) ================")
display(spearman)

# (B) Mean-difference tests across countries (ANOVA + Kruskal)
print("\n================ COUNTRY DIFFERENCES ================")
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

# (C) Baseline panel regression template (optional)
# Requires linearmodels; uses entity FE, year FE.
try:
    reg_df = DF2.dropna(subset=["ROA","NIM","Efficiency_Ratio","Fee_Income_Share"]).copy()
    reg_df = reg_df.set_index(["bank_id","year"])

    y = reg_df["ROA"]
    X = reg_df[["NIM","Efficiency_Ratio","Fee_Income_Share"]]
    X = sm.add_constant(X)

    mod = PanelOLS(y, X, entity_effects=True, time_effects=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)

    print("\n================ BASELINE PANELOLS (ROA) ================")
    print(res.summary)

except Exception as e:
    print("\nPanel regression skipped (missing package or data issue):", str(e))
