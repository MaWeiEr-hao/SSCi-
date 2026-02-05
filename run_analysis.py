import pandas as pd
import numpy as np
from scipy import stats
import os
import sys

# Shim for display()
if "display" not in globals():
    def display(*args):
        for arg in args:
            print(arg)
            print("-" * 20)

def load_data():
    filepath = "banks_reshaped.parquet"
    if os.path.exists(filepath):
        print(f"Loading data from {filepath}...")
        return pd.read_parquet(filepath)
    else:
        print(f"File {filepath} not found. Generating mock data...")
        return generate_mock_data()

def generate_mock_data():
    np.random.seed(42)
    n_banks = 200
    years = [2015, 2016, 2017, 2018, 2019, 2020]
    countries = ['US', 'UK', 'DE', 'JP', 'CN']

    data = []

    for bank_id in range(1, n_banks + 1):
        country = np.random.choice(countries)

        # Base values for each bank to ensure time-series variation around a mean
        base_nim = np.random.uniform(2, 5) * 1e6
        base_roa = np.random.uniform(0.5, 2) * 1e6
        base_roe = np.random.uniform(5, 15) * 1e6
        base_branches = np.random.randint(50, 1000)

        for year in years:
            # Add some random variation per year
            nim = base_nim + np.random.normal(0, 0.5 * 1e6)
            roa = base_roa + np.random.normal(0, 0.2 * 1e6)
            roe = base_roe + np.random.normal(0, 2 * 1e6)
            branches = int(base_branches * (1 + np.random.normal(0, 0.05)))

            # Efficiency Ratio components
            oper_revenues = np.random.uniform(100, 1000)
            # Efficiency ratio target ~ 50-80%
            ttl_oper_exps = oper_revenues * np.random.uniform(0.5, 0.8)

            fee_income_share = np.random.uniform(10, 40)

            # Deposits
            dep_demand = np.random.uniform(10, 40)
            dep_savings = np.random.uniform(10, 40)
            dep_time = np.random.uniform(10, 30)

            row = {
                "bank_id": bank_id,
                "year": year,
                "country": country,
                "Net_Int_margin": nim,
                "Return_on_Avg_assets_": roa,
                "Return_on_Avg_equity_": roe,
                "Ttl_Oper__Exps": ttl_oper_exps,
                "Oper__revenues": oper_revenues,
                "Branches": branches,
                "Fee_Income_Share": fee_income_share,
                "Dep_Demand_Share": dep_demand,
                "Dep_Savings_Share": dep_savings,
                "Dep_Time_Share": dep_time
            }
            data.append(row)

            # Update base branches for next year (drift)
            base_branches = branches

    return pd.DataFrame(data)

def preprocess_data(df):
    df = df.copy()

    # Scale variables
    # Memory: "divided by 1e6 to obtain percentages"
    df["NIM"] = df["Net_Int_margin"] / 1e6
    df["ROA"] = df["Return_on_Avg_assets_"] / 1e6
    df["ROE"] = df["Return_on_Avg_equity_"] / 1e6

    # Calculate Efficiency Ratio
    # Memory: "Efficiency_Ratio is calculated as (Ttl_Oper__Exps / Oper__revenues * 100)"
    df["Efficiency_Ratio"] = (df["Ttl_Oper__Exps"] / df["Oper__revenues"]) * 100

    # Calculate dBranches_pct
    # Memory: "dBranches_pct is calculated as the year-over-year percentage change in Branches grouped by bank_id"
    df = df.sort_values(by=["bank_id", "year"])
    df["dBranches_pct"] = df.groupby("bank_id")["Branches"].pct_change() * 100

    return df

def main():
    # Load raw data
    raw_df = load_data()

    # Preprocess
    DF = preprocess_data(raw_df)

    print(f"Data loaded and preprocessed. Shape: {DF.shape}")
    print("Columns:", DF.columns.tolist())

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

if __name__ == "__main__":
    main()
