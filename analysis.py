import pandas as pd
import numpy as np
from scipy import stats
import sys
import os

def load_and_prep_data(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return None

    try:
        df = pd.read_parquet(filepath)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return None

    # Filter for Germany, France, Italy (2010–2022)
    # Assuming columns are 'country' and 'year'
    required_cols = ['country', 'year']
    if not all(col in df.columns for col in required_cols):
         print(f"Error: Missing required columns for filtering: {required_cols}")
         return None

    countries = ['Germany', 'France', 'Italy']
    # Normalize country names if needed (e.g. title case)
    # But let's assume they match the memory description

    df_filtered = df[
        (df['country'].isin(countries)) &
        (df['year'] >= 2010) &
        (df['year'] <= 2022)
    ].copy()

    return df_filtered

def run_analysis(df):
    DF = df
    # ================== READY-TO-RUN TESTS (WITH SAFETY FILTERS) ==================
    # ... (paste user code here)

    # User code starts here
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
    # Check if columns exist before dropping
    subset_cols = [c for c in ["bank_id", "year", "country"] if c in DF2.columns]
    if len(subset_cols) < 3:
        print("Warning: One or more core identifiers (bank_id, year, country) missing.")

    DF2 = DF2.dropna(subset=subset_cols).copy()

    print("After plausibility screens (set to NaN, not dropped):")
    # Describe only columns that exist
    desc_cols = [c for c in ["NIM","ROA","ROE","Efficiency_Ratio"] if c in DF2.columns]
    if desc_cols:
        print(DF2[desc_cols].describe(percentiles=[.01,.5,.99]).T[["count","mean","50%","1%","99%"]])

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
    print(pearson) # Changed display() to print() as display is for notebooks

    print("\n================ CORRELATION (SPEARMAN) ================")
    print(spearman)

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

        # Using f-string for printing
        # Handling the dict comprehension in f-string might be tricky with quotes
        counts_dict = {k:len(vv) for k,vv in g.items()}
        print(f"{v:>18s} | ANOVA p={pval:.4g} | Kruskal p={pval_k:.4g} | Ns: {counts_dict}")

    # (C) Baseline panel regression template (optional)
    # Requires linearmodels; uses entity FE, year FE.
    try:
        from linearmodels.panel import PanelOLS
        import statsmodels.api as sm

        reg_cols = ["ROA","NIM","Efficiency_Ratio","Fee_Income_Share"]
        if not all(c in DF2.columns for c in reg_cols):
             print("\nPanel regression skipped: Missing required columns.")
        else:
            reg_df = DF2.dropna(subset=reg_cols).copy()
            # Ensure index is set correctly
            if "bank_id" in reg_df.columns and "year" in reg_df.columns:
                reg_df = reg_df.set_index(["bank_id","year"])

                y = reg_df["ROA"]
                X = reg_df[["NIM","Efficiency_Ratio","Fee_Income_Share"]]
                X = sm.add_constant(X)

                mod = PanelOLS(y, X, entity_effects=True, time_effects=True)
                res = mod.fit(cov_type="clustered", cluster_entity=True)

                print("\n================ BASELINE PANELOLS (ROA) ================")
                print(res.summary)
            else:
                print("\nPanel regression skipped: Missing bank_id or year columns for index.")

    except Exception as e:
        print("\nPanel regression skipped (missing package or data issue):", str(e))

if __name__ == "__main__":
    data_path = "banks_reshaped.parquet"
    print(f"Loading data from {data_path}...")
    df = load_and_prep_data(data_path)

    if df is not None:
        print(f"Data loaded. Shape: {df.shape}")
        run_analysis(df)
    else:
        print("Data loading failed.")
