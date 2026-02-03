import numpy as np
import pandas as pd
from scipy import stats
import os
import sys

# Shim for display()
def display(*args):
    for arg in args:
        print(arg)

def generate_mock_data():
    """
    Generates mock data for 200 banks over 6 years (2015-2020) with time-series variation.
    """
    print("Generating mock data...")
    np.random.seed(42)
    bank_ids = range(1, 201)
    years = range(2015, 2021)
    countries = ['US', 'UK', 'DE', 'JP', 'CN']

    data = []
    for bid in bank_ids:
        country = np.random.choice(countries)
        # Base values for this bank to ensure FE relevance
        # NIM scaled by 1e6. 2-4% -> 2,000,000 - 4,000,000
        base_nim = np.random.uniform(1.5, 4.5)
        # ROA scaled by 1e6. 0.5-1.5% -> 500,000 - 1,500,000
        base_roa = np.random.uniform(0.5, 1.5)
        # ROE scaled by 1e6. 5-15% -> 5,000,000 - 15,000,000
        base_roe = np.random.uniform(5, 15)
        # Efficiency Ratio ~ 50-80%
        base_eff = np.random.uniform(50, 80)

        # Initial branches
        branches = int(np.random.uniform(50, 500))

        for year in years:
            # Add some time variation and noise
            nim = base_nim + np.random.normal(0, 0.2)
            roa = base_roa + np.random.normal(0, 0.1)
            roe = base_roe + np.random.normal(0, 1.0)

            # Construct raw scaled values
            net_int_margin = nim * 1e6
            roa_raw = roa * 1e6
            roe_raw = roe * 1e6

            oper_rev = np.random.uniform(100, 1000) * 1e6
            # eff = exps/rev * 100 -> exps = eff/100 * rev
            eff_ratio = base_eff + np.random.normal(0, 2)
            ttl_oper_exps = (eff_ratio / 100) * oper_rev

            # Branches growth/decline
            branches = int(branches * (1 + np.random.normal(0, 0.02)))
            if branches < 1: branches = 1

            fee_share = np.random.uniform(10, 40)
            dep_demand = np.random.uniform(10, 50)
            dep_savings = np.random.uniform(10, 50)
            dep_time = np.random.uniform(10, 50)

            data.append({
                'bank_id': bid,
                'year': year,
                'country': country,
                'Net_Int_margin': net_int_margin,
                'Return_on_Avg_assets_': roa_raw,
                'Return_on_Avg_equity_': roe_raw,
                'Ttl_Oper__Exps': ttl_oper_exps,
                'Oper__revenues': oper_rev,
                'Branches': branches,
                'Fee_Income_Share': fee_share,
                'Dep_Demand_Share': dep_demand,
                'Dep_Savings_Share': dep_savings,
                'Dep_Time_Share': dep_time
            })

    return pd.DataFrame(data)

def load_and_process_data():
    filepath = 'banks_reshaped.parquet'
    if os.path.exists(filepath):
        print(f"Loading {filepath}...")
        df = pd.read_parquet(filepath)
    else:
        print(f"{filepath} not found. Using mock data.")
        df = generate_mock_data()

    # Preprocessing
    # Scale variables

    if 'Net_Int_margin' in df.columns:
        df['NIM'] = df['Net_Int_margin'] / 1e6

    if 'Return_on_Avg_assets_' in df.columns:
        df['ROA'] = df['Return_on_Avg_assets_'] / 1e6

    if 'Return_on_Avg_equity_' in df.columns:
        df['ROE'] = df['Return_on_Avg_equity_'] / 1e6

    if 'Ttl_Oper__Exps' in df.columns and 'Oper__revenues' in df.columns:
        # Avoid division by zero if any
        df['Efficiency_Ratio'] = (df['Ttl_Oper__Exps'] / df['Oper__revenues'].replace(0, np.nan)) * 100

    # dBranches_pct
    if 'Branches' in df.columns:
        df = df.sort_values(['bank_id', 'year'])
        df['dBranches_pct'] = df.groupby('bank_id')['Branches'].pct_change() * 100

    return df

if __name__ == "__main__":
    DF = load_and_process_data()

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
        # Set index for PanelOLS
        if 'bank_id' in reg_df.columns and 'year' in reg_df.columns:
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
