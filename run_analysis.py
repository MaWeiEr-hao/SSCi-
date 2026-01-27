import numpy as np
import pandas as pd
from scipy import stats
import sys

# Shim display for console output
def display(obj):
    print(obj)

def generate_mock_data():
    print("Generating mock data (200 banks, 2015-2020)...")
    np.random.seed(42)
    n_banks = 200
    years = range(2015, 2021)
    countries = ['US', 'UK', 'DE', 'JP', 'CN']

    records = []
    for bank_id in range(1, n_banks + 1):
        country = np.random.choice(countries)
        # Base values for this bank to create some entity persistence
        base_nim = np.random.normal(3, 0.5)
        base_roa = np.random.normal(1, 0.3)
        base_eff = np.random.normal(60, 10)
        base_branches = np.random.randint(50, 1000)

        for year in years:
            # Add time variation and random noise
            nim = base_nim + np.random.normal(0, 0.2)
            roa = base_roa + np.random.normal(0, 0.1)
            roe = roa * np.random.uniform(8, 12) # roughly correlated
            eff = base_eff + np.random.normal(0, 5)
            fee_share = np.random.normal(20, 5)

            # Branches evolve slowly
            branches = int(base_branches * np.random.uniform(0.95, 1.05))
            base_branches = branches

            records.append({
                'bank_id': bank_id,
                'year': year,
                'country': country,
                'NIM': nim,
                'ROA': roa,
                'ROE': roe,
                'Efficiency_Ratio': eff,
                'Fee_Income_Share': fee_share,
                'Dep_Demand_Share': np.random.uniform(10, 40),
                'Dep_Savings_Share': np.random.uniform(10, 40),
                'Dep_Time_Share': np.random.uniform(10, 40),
                'Branches': branches
            })

    df = pd.DataFrame(records)

    # Calculate dBranches_pct
    df = df.sort_values(['bank_id', 'year'])
    df['dBranches_pct'] = df.groupby('bank_id')['Branches'].pct_change() * 100

    return df

# Main execution block
if __name__ == "__main__":
    # Check for existing data, else mock
    try:
        DF = pd.read_parquet('banks_reshaped.parquet')
        print("Loaded banks_reshaped.parquet")
    except FileNotFoundError:
        DF = generate_mock_data()

    # ================== READY-TO-RUN TESTS (WITH SAFETY FILTERS) ==================
    # (Code provided by user)

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
        # Ensure MultiIndex for PanelOLS
        if "bank_id" in reg_df.columns and "year" in reg_df.columns:
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
