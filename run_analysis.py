import numpy as np
import pandas as pd
from scipy import stats
import os

try:
    from linearmodels.panel import PanelOLS
    import statsmodels.api as sm
    HAS_LINEARMODELS = True
except ImportError:
    HAS_LINEARMODELS = False
    print("linearmodels not installed or failed to import. Regression steps might be skipped.")

def load_or_generate_data():
    """Loads banks_reshaped.parquet or generates mock data if missing."""
    file_path = "banks_reshaped.parquet"
    if os.path.exists(file_path):
        print(f"Loading data from {file_path}...")
        return pd.read_parquet(file_path)
    else:
        print(f"{file_path} not found. Generating mock data for testing...")

        np.random.seed(42)
        n_banks = 50
        n_years = 11
        years = np.arange(2010, 2021)
        countries = ['US', 'UK', 'DE', 'JP']

        data = []
        for bank_id in range(1, n_banks + 1):
            country = np.random.choice(countries)
            base_nim = np.random.normal(3, 0.5)
            base_roa = np.random.normal(1, 0.2)
            base_roe = np.random.normal(10, 2)
            base_eff = np.random.normal(60, 5)

            for year in years:
                # Add some random variation per year
                row = {
                    "bank_id": bank_id,
                    "year": year,
                    "country": country,
                    "NIM": base_nim + np.random.normal(0, 0.2),
                    "ROA": base_roa + np.random.normal(0, 0.1),
                    "ROE": base_roe + np.random.normal(0, 1),
                    "Efficiency_Ratio": base_eff + np.random.normal(0, 2),
                    "Fee_Income_Share": np.random.normal(20, 5),
                    "Dep_Demand_Share": np.random.normal(30, 5),
                    "Dep_Savings_Share": np.random.normal(30, 5),
                    "Dep_Time_Share": np.random.normal(30, 5),
                    "Branches": int(np.random.normal(100, 20)),
                    "dBranches_pct": np.random.normal(0, 2)
                }
                data.append(row)

        df = pd.DataFrame(data)

        # Introduce some "pathological" outliers to test filters
        # e.g. one very high ROE
        df.loc[0, "ROE"] = 500
        # one negative NIM
        df.loc[1, "NIM"] = -25

        return df

def run_user_analysis(DF):
    """Runs the user-provided analysis steps."""

    # User's Code Start
    # ================== READY-TO-RUN TESTS (WITH SAFETY FILTERS) ==================
    # (Imports are handled at top level)

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
    print(pearson) # Changed display() to print()

    print("\n================ CORRELATION (SPEARMAN) ================")
    print(spearman) # Changed display() to print()

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
        # Imports handled at top level, but checking here again
        if not HAS_LINEARMODELS:
            raise ImportError("linearmodels not available")

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

    # User's Code End

if __name__ == "__main__":
    DF = load_or_generate_data()
    run_user_analysis(DF)
