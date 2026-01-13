import numpy as np
import pandas as pd
from scipy import stats
import os

# Shim for display() if not in a notebook
try:
    display
except NameError:
    def display(obj):
        print(obj)

def generate_mock_data():
    """Generates mock data for testing purposes."""
    np.random.seed(42)
    n_rows = 500
    countries = ['US', 'UK', 'DE', 'FR', 'JP', 'CA', 'AU']

    data = {
        'bank_id': np.random.randint(1, 101, n_rows),
        'year': np.random.randint(2015, 2021, n_rows),
        'country': np.random.choice(countries, n_rows),

        # Financial ratios (normally distributed mostly)
        'NIM': np.random.normal(3.0, 1.0, n_rows),
        'ROA': np.random.normal(1.0, 0.5, n_rows),
        'ROE': np.random.normal(10.0, 5.0, n_rows),
        'Efficiency_Ratio': np.random.normal(60.0, 15.0, n_rows),
        'Fee_Income_Share': np.random.normal(25.0, 10.0, n_rows),

        # Deposit shares
        'Dep_Demand_Share': np.random.uniform(0, 50, n_rows),
        'Dep_Savings_Share': np.random.uniform(0, 50, n_rows),
        'Dep_Time_Share': np.random.uniform(0, 50, n_rows),

        # Other
        'Branches': np.random.randint(5, 500, n_rows),
        'dBranches_pct': np.random.normal(0, 5, n_rows),
    }

    df = pd.DataFrame(data)

    # Add some outliers to test the sanity screens
    df.loc[0, 'ROA'] = 50  # Should be caught by (-20, 20)
    df.loc[1, 'NIM'] = -30 # Should be caught by (-20, 20)

    return df

def load_data():
    file_path = 'banks_reshaped.parquet'
    if os.path.exists(file_path):
        print(f"Loading {file_path}...")
        df = pd.read_parquet(file_path)

        # Data Handling based on memory instructions
        # Scale by 1e6 if raw columns exist and processed ones don't
        if 'Net_Int_margin' in df.columns and 'NIM' not in df.columns:
            df['NIM'] = df['Net_Int_margin'] / 1e6
        if 'Return_on_Avg_assets_' in df.columns and 'ROA' not in df.columns:
            df['ROA'] = df['Return_on_Avg_assets_'] / 1e6
        if 'Return_on_Avg_equity_' in df.columns and 'ROE' not in df.columns:
            df['ROE'] = df['Return_on_Avg_equity_'] / 1e6

        # Efficiency Ratio calculation
        if 'Efficiency_Ratio' not in df.columns:
            if 'Ttl_Oper__Exps' in df.columns and 'Oper__revenues' in df.columns:
                # Avoid division by zero
                df['Efficiency_Ratio'] = df['Ttl_Oper__Exps'] / df['Oper__revenues'].replace(0, np.nan) * 100

        return df
    else:
        print("Dataset not found. Generating mock data...")
        return generate_mock_data()

# ================== MAIN EXECUTION ==================

if __name__ == "__main__":
    # Load Data
    DF = load_data()
    print(f"Data loaded. Shape: {DF.shape}")
    print("Columns:", DF.columns.tolist())

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
            # Using boolean indexing for assignment to avoid SettingWithCopy warnings if relevant,
            # though DF2 is a fresh copy.
            mask = ~DF2[v].between(lo, hi)
            DF2.loc[mask, v] = np.nan

    # Also drop bank-years with missing core identifiers
    # Check if these columns exist before dropping to avoid errors in mock data if we messed up
    subset_cols = [c for c in ["bank_id", "year", "country"] if c in DF2.columns]
    DF2 = DF2.dropna(subset=subset_cols).copy()

    print("After plausibility screens (set to NaN, not dropped):")
    # Helper to avoid error if columns missing
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

        reg_cols = ["ROA","NIM","Efficiency_Ratio","Fee_Income_Share"]
        if all(c in DF2.columns for c in reg_cols):
            reg_df = DF2.dropna(subset=reg_cols).copy()
            reg_df = reg_df.set_index(["bank_id","year"])

            y = reg_df["ROA"]
            X = reg_df[["NIM","Efficiency_Ratio","Fee_Income_Share"]]
            X = sm.add_constant(X)

            mod = PanelOLS(y, X, entity_effects=True, time_effects=True)
            res = mod.fit(cov_type="clustered", cluster_entity=True)

            print("\n================ BASELINE PANELOLS (ROA) ================")
            print(res.summary)
        else:
             print("\nSkipping regression: missing required columns.")

    except Exception as e:
        print("\nPanel regression skipped (missing package or data issue):", str(e))
