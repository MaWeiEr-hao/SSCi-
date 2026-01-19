import numpy as np
import pandas as pd
from scipy import stats
import sys

# Shim for display() if running as a script
if 'display' not in globals():
    def display(*args):
        for arg in args:
            print(arg)

# ================== MOCK DATA GENERATION ==================
# Fallback mechanism as described in memory
def generate_mock_data():
    print("Generating mock data...")
    np.random.seed(42)
    n_banks = 200
    n_years = 10
    years = list(range(2010, 2010 + n_years)) * n_banks
    bank_ids = [f"BANK_{i:03d}" for i in range(n_banks) for _ in range(n_years)]
    countries = ["Country_A", "Country_B", "Country_C", "Country_D"]
    # Assign countries to banks
    bank_country_map = {f"BANK_{i:03d}": np.random.choice(countries) for i in range(n_banks)}
    country_col = [bank_country_map[bid] for bid in bank_ids]

    n_rows = len(bank_ids)

    data = {
        "bank_id": bank_ids,
        "year": years,
        "country": country_col,
        "NIM": np.random.normal(3.0, 1.0, n_rows), # Net Interest Margin %
        "ROA": np.random.normal(1.0, 0.5, n_rows), # Return on Assets %
        "ROE": np.random.normal(10.0, 5.0, n_rows), # Return on Equity %
        "Efficiency_Ratio": np.random.normal(60.0, 15.0, n_rows), # Cost to Income
        "Fee_Income_Share": np.random.normal(20.0, 10.0, n_rows),
        "Dep_Demand_Share": np.random.normal(30.0, 10.0, n_rows),
        "Dep_Savings_Share": np.random.normal(30.0, 10.0, n_rows),
        "Dep_Time_Share": np.random.normal(30.0, 10.0, n_rows),
        "Branches": np.random.randint(10, 1000, n_rows),
        "dBranches_pct": np.random.normal(0, 5, n_rows) # percent change in branches
    }

    df = pd.DataFrame(data)

    # Add some outliers to test the screens
    # NIM outlier
    if len(df) > 0:
        df.loc[0, "NIM"] = 50.0
        # ROE outlier
        df.loc[1, "ROE"] = 500.0
        # Negative Efficiency (impossible usually but good for test)
        df.loc[2, "Efficiency_Ratio"] = -5.0

    return df

# Try loading data, else mock
try:
    DF = pd.read_parquet("banks_reshaped.parquet")
    print("Loaded banks_reshaped.parquet")
except FileNotFoundError:
    print("banks_reshaped.parquet not found. Using mock data.")
    DF = generate_mock_data()
except Exception as e:
    print(f"Error loading data: {e}. Using mock data.")
    DF = generate_mock_data()

# ================== READY-TO-RUN TESTS (WITH SAFETY FILTERS) ==================
# (Pasted from user request)

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
# Fix: The user code uses .T[["count"...]] but describe().T might reorder or user assumes column names.
# Pandas describe() returns rows like count, mean, std, min, ...
# So .T puts stats as columns.
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
    # Drop NaNs for the specific variable before grouping
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
    if reg_df.empty:
        print("Regression dataframe is empty after dropping NaNs.")
    else:
        # Check if MultiIndex is already set or duplicates exist
        # PanelOLS requires index [Entity, Time]
        # We need to ensure uniqueness.
        # User code assumes simple set_index. I will add check.
        if reg_df.duplicated(subset=["bank_id", "year"]).any():
            print("Duplicates found in bank_id/year, handling by mean aggregation...")
            reg_df = reg_df.groupby(["bank_id", "year"]).mean(numeric_only=True).reset_index()

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
