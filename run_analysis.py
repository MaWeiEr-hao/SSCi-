import numpy as np
import pandas as pd
from scipy import stats
import os

# Shim for display() if not in a notebook environment
def display(obj):
    if isinstance(obj, pd.DataFrame):
        print(obj.to_string())
    else:
        print(obj)

def generate_mock_data():
    print("Generating mock data...")
    np.random.seed(42)
    n_banks = 200
    years = [2015, 2016, 2017, 2018, 2019, 2020]
    countries = ['CountryA', 'CountryB', 'CountryC', 'CountryD', 'CountryE']

    data = []

    for bank_id in range(1, n_banks + 1):
        country = np.random.choice(countries)

        # Base values for this bank
        base_nim = np.random.uniform(0.02, 0.05) * 1e6
        base_roa = np.random.uniform(0.005, 0.02) * 1e6
        base_roe = np.random.uniform(0.05, 0.15) * 1e6
        base_rev = np.random.uniform(500000, 2000000)
        base_exp_ratio = np.random.uniform(0.4, 0.8)
        base_fee_share = np.random.uniform(10, 40)
        base_branches = np.random.randint(20, 500)

        # Deposit shares base
        base_dep_demand = np.random.uniform(10, 50)
        base_dep_savings = np.random.uniform(10, 50)
        base_dep_time = np.random.uniform(10, 50)

        for year in years:
            # Add time-series variation (random walk / noise)
            nim = base_nim + np.random.normal(0, 0.005 * 1e6)
            roa = base_roa + np.random.normal(0, 0.002 * 1e6)
            roe = base_roe + np.random.normal(0, 0.02 * 1e6)

            rev = base_rev * (1 + np.random.normal(0, 0.05))
            exp = rev * (base_exp_ratio + np.random.normal(0, 0.02))

            fee_share = base_fee_share + np.random.normal(0, 2)

            branches = int(base_branches * (1 + np.random.normal(0, 0.02)))
            base_branches = branches # Evolve branches

            dep_demand = base_dep_demand + np.random.normal(0, 2)
            dep_savings = base_dep_savings + np.random.normal(0, 2)
            dep_time = base_dep_time + np.random.normal(0, 2)

            data.append({
                'bank_id': bank_id,
                'year': year,
                'country': country,
                'Net_Int_margin': nim,
                'Return_on_Avg_assets_': roa,
                'Return_on_Avg_equity_': roe,
                'Ttl_Oper__Exps': exp,
                'Oper__revenues': rev,
                'Fee_Income_Share': fee_share,
                'Dep_Demand_Share': dep_demand,
                'Dep_Savings_Share': dep_savings,
                'Dep_Time_Share': dep_time,
                'Branches': branches
            })

    return pd.DataFrame(data)

def preprocess_data(df):
    print("Preprocessing data...")
    # Map columns as per requirements
    # NIM, ROA, ROE are scaled by 1e6 in raw data
    if 'Net_Int_margin' in df.columns:
        df['NIM'] = df['Net_Int_margin'] / 1e6
    if 'Return_on_Avg_assets_' in df.columns:
        df['ROA'] = df['Return_on_Avg_assets_'] / 1e6
    if 'Return_on_Avg_equity_' in df.columns:
        df['ROE'] = df['Return_on_Avg_equity_'] / 1e6

    # Efficiency Ratio calculation
    if 'Ttl_Oper__Exps' in df.columns and 'Oper__revenues' in df.columns:
        df['Efficiency_Ratio'] = (df['Ttl_Oper__Exps'] / df['Oper__revenues']) * 100

    # dBranches_pct calculation
    if 'Branches' in df.columns and 'bank_id' in df.columns:
        df['dBranches_pct'] = df.sort_values(['bank_id', 'year']).groupby('bank_id')['Branches'].pct_change() * 100

    return df

def main():
    # Load or generate data
    data_path = 'banks_reshaped.parquet'
    if os.path.exists(data_path):
        print(f"Loading data from {data_path}...")
        try:
            DF = pd.read_parquet(data_path)
        except Exception as e:
            print(f"Error loading parquet: {e}. Falling back to mock data.")
            DF = generate_mock_data()
    else:
        print(f"{data_path} not found. Falling back to mock data.")
        DF = generate_mock_data()

    DF = preprocess_data(DF)

    # ================== READY-TO-RUN TESTS (WITH SAFETY FILTERS) ==================
    # [User Snippet starts here]

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
    cols_to_check = ["bank_id", "year", "country"]
    # Check if these columns exist before dropping
    existing_cols = [c for c in cols_to_check if c in DF2.columns]
    if len(existing_cols) == 3:
        DF2 = DF2.dropna(subset=existing_cols).copy()
    else:
        print(f"Warning: Missing core identifiers in DataFrame: {set(cols_to_check) - set(existing_cols)}")

    print("After plausibility screens (set to NaN, not dropped):")
    stats_cols = ["NIM","ROA","ROE","Efficiency_Ratio"]
    existing_stats_cols = [c for c in stats_cols if c in DF2.columns]
    if existing_stats_cols:
        print(DF2[existing_stats_cols].describe(percentiles=[.01,.5,.99]).T[["count","mean","50%","1%","99%"]])

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

        req_cols = ["ROA","NIM","Efficiency_Ratio","Fee_Income_Share"]
        if all(c in DF2.columns for c in req_cols):
            reg_df = DF2.dropna(subset=req_cols).copy()
            # Ensure index is set for PanelOLS
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
                 print("\nPanel regression skipped: 'bank_id' or 'year' missing.")
        else:
            print(f"\nPanel regression skipped: Missing required columns {set(req_cols) - set(DF2.columns)}")

    except Exception as e:
        print("\nPanel regression skipped (missing package or data issue):", str(e))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
