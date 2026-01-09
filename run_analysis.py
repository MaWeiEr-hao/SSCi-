import numpy as np
import pandas as pd
from scipy import stats
import os

# Set display for non-interactive environment
def display(obj):
    print(obj)

# --- 0) Data Loading / Mock Data ---

def generate_mock_data(n_banks=50, n_years=10):
    np.random.seed(42)
    bank_ids = [f'BANK_{i:03d}' for i in range(n_banks)]
    years = range(2010, 2010 + n_years)
    countries = ['US', 'UK', 'DE', 'FR', 'JP']

    data = []
    for bank in bank_ids:
        country = np.random.choice(countries)
        for year in years:
            row = {
                'bank_id': bank,
                'year': year,
                'country': country,
                'NIM': np.random.normal(3, 1),
                'ROA': np.random.normal(1, 0.5),
                'ROE': np.random.normal(10, 5),
                'Efficiency_Ratio': np.random.normal(60, 10),
                'Fee_Income_Share': np.random.normal(20, 5),
                'Dep_Demand_Share': np.random.normal(30, 10),
                'Dep_Savings_Share': np.random.normal(40, 10),
                'Dep_Time_Share': np.random.normal(30, 10),
                'Branches': np.random.randint(10, 1000),
            }
            # Add some outliers
            if np.random.rand() < 0.01:
                row['ROA'] = 50 # Outlier
            if np.random.rand() < 0.01:
                row['Efficiency_Ratio'] = -5 # Invalid

            data.append(row)

    df = pd.DataFrame(data)
    # Calculate dBranches_pct (pct change in branches)
    df = df.sort_values(['bank_id', 'year'])
    df['dBranches_pct'] = df.groupby('bank_id')['Branches'].pct_change() * 100

    return df

if os.path.exists('banks_reshaped.parquet'):
    print("Loading data from banks_reshaped.parquet...")
    DF = pd.read_parquet('banks_reshaped.parquet')
else:
    print("banks_reshaped.parquet not found. Generating mock data...")
    DF = generate_mock_data()

# ================== USER PROVIDED CODE STARTS HERE ==================

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
print(DF2[["NIM","ROA","ROE","Efficiency_Ratio"]].describe(percentiles=[.01,.5,.99]).T[["count","mean","50%","1%","99%"]])

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

except Exception as e:
    print("\nPanel regression skipped (missing package or data issue):", str(e))
