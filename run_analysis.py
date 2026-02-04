import numpy as np
import pandas as pd
from scipy import stats
import sys
import os

# Shim for display() if not in IPython
if 'display' not in locals() and 'display' not in globals():
    def display(obj):
        print(obj)

def generate_mock_data():
    print("Generating mock data...")
    n_banks = 200
    years = range(2015, 2021)
    countries = ['US', 'UK', 'DE', 'JP', 'CN']

    bank_ids = range(1, n_banks + 1)

    # Assign country to each bank
    bank_countries = np.random.choice(countries, n_banks)
    bank_country_map = dict(zip(bank_ids, bank_countries))

    data = []

    # Base values for random walk
    # Use dictionary to store current state for each bank
    current_states = {}

    for bank_id in bank_ids:
        # Initialize base state for this bank
        current_states[bank_id] = {
            'Net_Int_margin': np.random.uniform(1e6, 5e6), # 1% to 5% * 1e6
            'Return_on_Avg_assets_': np.random.uniform(0.5e6, 2e6), # 0.5% to 2% * 1e6
            'Return_on_Avg_equity_': np.random.uniform(5e6, 20e6), # 5% to 20% * 1e6
            'Branches': np.random.randint(10, 1000),
            'Ttl_Oper__Exps': np.random.uniform(100, 1000),
            'Oper__revenues': np.random.uniform(150, 1500),
            'Fee_Income_Share': np.random.uniform(10, 50),
            'Dep_Demand_Share': np.random.uniform(10, 50),
            'Dep_Savings_Share': np.random.uniform(10, 50),
            'Dep_Time_Share': np.random.uniform(10, 50),
        }

    for year in years:
        for bank_id in bank_ids:
            # Update state with random walk
            state = current_states[bank_id]

            # Apply random shocks
            state['Net_Int_margin'] += np.random.normal(0, 0.2e6)
            state['Return_on_Avg_assets_'] += np.random.normal(0, 0.1e6)
            state['Return_on_Avg_equity_'] += np.random.normal(0, 1e6)
            state['Branches'] = max(1, int(state['Branches'] * np.random.normal(1.0, 0.05)))

            # Efficiency components
            state['Oper__revenues'] += np.random.normal(0, 50)
            if state['Oper__revenues'] < 10: state['Oper__revenues'] = 10

            state['Ttl_Oper__Exps'] = state['Oper__revenues'] * np.random.uniform(0.5, 0.8)

            # Shares
            state['Fee_Income_Share'] = np.clip(state['Fee_Income_Share'] + np.random.normal(0, 2), 0, 100)
            state['Dep_Demand_Share'] = np.clip(state['Dep_Demand_Share'] + np.random.normal(0, 2), 0, 100)
            state['Dep_Savings_Share'] = np.clip(state['Dep_Savings_Share'] + np.random.normal(0, 2), 0, 100)
            state['Dep_Time_Share'] = np.clip(state['Dep_Time_Share'] + np.random.normal(0, 2), 0, 100)

            row = {
                'bank_id': bank_id,
                'year': year,
                'country': bank_country_map[bank_id],
                'Net_Int_margin': state['Net_Int_margin'],
                'Return_on_Avg_assets_': state['Return_on_Avg_assets_'],
                'Return_on_Avg_equity_': state['Return_on_Avg_equity_'],
                'Branches': state['Branches'],
                'Ttl_Oper__Exps': state['Ttl_Oper__Exps'],
                'Oper__revenues': state['Oper__revenues'],
                'Fee_Income_Share': state['Fee_Income_Share'],
                'Dep_Demand_Share': state['Dep_Demand_Share'],
                'Dep_Savings_Share': state['Dep_Savings_Share'],
                'Dep_Time_Share': state['Dep_Time_Share']
            }
            data.append(row)

    df = pd.DataFrame(data)
    return df

def load_or_generate_data():
    filepath = 'banks_reshaped.parquet'
    if os.path.exists(filepath):
        print(f"Loading data from {filepath}...")
        try:
            return pd.read_parquet(filepath)
        except Exception as e:
            print(f"Error loading parquet: {e}")
            return generate_mock_data()
    else:
        print(f"{filepath} not found.")
        return generate_mock_data()

def preprocess_data(df):
    print("Preprocessing data...")
    # Scaling
    # Net_Int_margin, Return_on_Avg_assets_, Return_on_Avg_equity_ are scaled by 1,000,000
    if 'Net_Int_margin' in df.columns:
        df['NIM'] = df['Net_Int_margin'] / 1e6
    if 'Return_on_Avg_assets_' in df.columns:
        df['ROA'] = df['Return_on_Avg_assets_'] / 1e6
    if 'Return_on_Avg_equity_' in df.columns:
        df['ROE'] = df['Return_on_Avg_equity_'] / 1e6

    # Efficiency Ratio
    if 'Ttl_Oper__Exps' in df.columns and 'Oper__revenues' in df.columns:
        # Avoid division by zero
        df['Efficiency_Ratio'] = (df['Ttl_Oper__Exps'] / df['Oper__revenues'].replace(0, np.nan)) * 100

    # dBranches_pct
    if 'Branches' in df.columns:
        df = df.sort_values(['bank_id', 'year'])
        df['dBranches_pct'] = df.groupby('bank_id')['Branches'].pct_change() * 100

    return df

def main():
    DF = load_or_generate_data()
    DF = preprocess_data(DF)

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
    if not DF2.empty:
        check_cols = ["NIM","ROA","ROE","Efficiency_Ratio"]
        check_cols = [c for c in check_cols if c in DF2.columns]
        if check_cols:
            print(DF2[check_cols].describe(percentiles=[.01,.5,.99]).T[["count","mean","50%","1%","99%"]])

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
            reg_df = reg_df.set_index(["bank_id","year"])

            y = reg_df["ROA"]
            X = reg_df[["NIM","Efficiency_Ratio","Fee_Income_Share"]]
            X = sm.add_constant(X)

            mod = PanelOLS(y, X, entity_effects=True, time_effects=True)
            res = mod.fit(cov_type="clustered", cluster_entity=True)

            print("\n================ BASELINE PANELOLS (ROA) ================")
            print(res.summary)
        else:
            print("\nRegression skipped: missing columns.")

    except Exception as e:
        print("\nPanel regression skipped (missing package or data issue):", str(e))

if __name__ == "__main__":
    main()
