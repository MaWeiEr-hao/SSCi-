import numpy as np
import pandas as pd
from scipy import stats
import os
import sys

# ---------- 0) Setup & Data Loading ----------

def get_data():
    parquet_path = "banks_reshaped.parquet"
    if os.path.exists(parquet_path):
        print(f"Loading data from {parquet_path}...")
        df = pd.read_parquet(parquet_path)
    else:
        print(f"Warning: {parquet_path} not found. Generating mock data.")
        # Generate mock data
        np.random.seed(42)
        n_banks = 50
        n_years = 10
        countries = ['US', 'UK', 'DE', 'FR', 'JP']

        data = []
        for bank_id in range(n_banks):
            country = np.random.choice(countries)
            for year in range(2010, 2010 + n_years):
                # Random raw values
                # Scaling by 1e6 as per memory instructions for raw data
                net_int_margin = np.random.normal(2.5, 0.5) * 1e6
                return_on_avg_assets = np.random.normal(1.0, 0.2) * 1e6
                return_on_avg_equity = np.random.normal(10.0, 2.0) * 1e6

                # Efficiency ratio components
                oper_revenues = np.random.uniform(100, 1000)
                efficiency_target = np.random.uniform(40, 80)
                ttl_oper_exps = oper_revenues * (efficiency_target / 100)

                fee_income_share = np.random.uniform(10, 40)
                dep_demand_share = np.random.uniform(20, 50)
                dep_savings_share = np.random.uniform(20, 40)
                dep_time_share = np.random.uniform(10, 30)

                branches = np.random.randint(50, 500)
                dbranches_pct = np.random.normal(0, 5)

                data.append({
                    'bank_id': bank_id,
                    'year': year,
                    'country': country,
                    'Net_Int_margin': net_int_margin,
                    'Return_on_Avg_assets_': return_on_avg_assets,
                    'Return_on_Avg_equity_': return_on_avg_equity,
                    'Ttl_Oper__Exps': ttl_oper_exps,
                    'Oper__revenues': oper_revenues,
                    'Fee_Income_Share': fee_income_share,
                    'Dep_Demand_Share': dep_demand_share,
                    'Dep_Savings_Share': dep_savings_share,
                    'Dep_Time_Share': dep_time_share,
                    'Branches': branches,
                    'dBranches_pct': dbranches_pct
                })
        df = pd.DataFrame(data)
    return df

def preprocess_data(df):
    # Data Handling as per memory
    if 'NIM' not in df.columns and 'Net_Int_margin' in df.columns:
        df['NIM'] = df['Net_Int_margin'] / 1e6

    if 'ROA' not in df.columns and 'Return_on_Avg_assets_' in df.columns:
        df['ROA'] = df['Return_on_Avg_assets_'] / 1e6

    if 'ROE' not in df.columns and 'Return_on_Avg_equity_' in df.columns:
        df['ROE'] = df['Return_on_Avg_equity_'] / 1e6

    if 'Efficiency_Ratio' not in df.columns and 'Ttl_Oper__Exps' in df.columns and 'Oper__revenues' in df.columns:
        df['Efficiency_Ratio'] = (df['Ttl_Oper__Exps'] / df['Oper__revenues']) * 100

    return df

# Helper for display compatibility
def display(obj):
    print(obj)

# Main Execution Flow
if __name__ == "__main__":
    DF = get_data()
    DF = preprocess_data(DF)

    # ================== READY-TO-RUN TESTS (WITH SAFETY FILTERS) ==================
    # (Pasted user code below)

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
