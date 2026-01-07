import numpy as np
import pandas as pd
from scipy import stats
import sys
import argparse

def load_and_preprocess_data(filepath):
    """
    Loads data from parquet and applies necessary scaling.
    """
    try:
        df = pd.read_parquet(filepath)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None

    # Scaling and Renaming based on historical context (Memory),
    # but strictly applied only to align units for the analysis.
    # Net_Int_margin, Return_on_Avg_assets_, Return_on_Avg_equity_ are scaled by 1,000,000
    # and must be divided by 1e6 to obtain percentages (or ratios).

    if 'Net_Int_margin' in df.columns:
        df['NIM'] = df['Net_Int_margin'] / 1e6

    if 'Return_on_Avg_assets_' in df.columns:
        df['ROA'] = df['Return_on_Avg_assets_'] / 1e6

    if 'Return_on_Avg_equity_' in df.columns:
        df['ROE'] = df['Return_on_Avg_equity_'] / 1e6

    # Efficiency_ratio should be calculated from components (Ttl_Oper__Exps / Oper__revenues)
    if 'Ttl_Oper__Exps' in df.columns and 'Oper__revenues' in df.columns:
        # Avoid division by zero
        df['Efficiency_Ratio'] = (df['Ttl_Oper__Exps'] / df['Oper__revenues'].replace(0, np.nan)) * 100

    return df

def run_analysis(DF):
    """
    Executes the user-provided analysis and tests.
    """
    print("Starting analysis...")

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
    if subset_cols:
        DF2 = DF2.dropna(subset=subset_cols).copy()

    print("After plausibility screens (set to NaN, not dropped):")
    desc_cols = ["NIM","ROA","ROE","Efficiency_Ratio"]
    desc_cols = [c for c in desc_cols if c in DF2.columns]
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
    if not corr_df.empty:
        pearson = corr_df.corr(method="pearson")
        spearman = corr_df.corr(method="spearman")

        print("\n================ CORRELATION (PEARSON) ================")
        print(pearson)

        print("\n================ CORRELATION (SPEARMAN) ================")
        print(spearman)

    # (B) Mean-difference tests across countries (ANOVA + Kruskal)
    print("\n================ COUNTRY DIFFERENCES ================")
    if "country" in DF2.columns:
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

        req_cols = ["ROA","NIM","Efficiency_Ratio","Fee_Income_Share", "bank_id", "year"]
        if all(c in DF2.columns for c in req_cols):
            reg_df = DF2.dropna(subset=["ROA","NIM","Efficiency_Ratio","Fee_Income_Share"]).copy()
            reg_df = reg_df.set_index(["bank_id","year"])

            y = reg_df["ROA"]
            X = reg_df[["NIM","Efficiency_Ratio","Fee_Income_Share"]]
            X = sm.add_constant(X)

            mod = PanelOLS(y, X, entity_effects=True, time_effects=True)
            res = mod.fit(cov_type="clustered", cluster_entity=True)

            print("\n================ BASELINE PANELOLS (ROA) ================")
            print(res.summary)
        else:
            print("\nSkipping Panel Regression: Missing required columns.")

    except ImportError:
        print("\nPanel regression skipped: linearmodels not installed.")
    except Exception as e:
        print("\nPanel regression skipped (error):", str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run bank data analysis.")
    parser.add_argument("filepath", nargs="?", default="banks_reshaped.parquet", help="Path to the parquet file.")
    args = parser.parse_args()

    df = load_and_preprocess_data(args.filepath)
    if df is not None:
        run_analysis(df)
