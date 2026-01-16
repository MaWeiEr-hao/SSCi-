import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

# Try importing linearmodels, handle if missing (though we installed it)
try:
    from linearmodels.panel import PanelOLS
except ImportError:
    PanelOLS = None
    print("Warning: linearmodels not installed. Panel regression will be skipped.")

# Shim for display()
def display(*args):
    for arg in args:
        print(arg)
        print("-" * 20)

def generate_mock_data(n_banks=50, years=range(2010, 2021)):
    np.random.seed(42)
    data = []
    countries = ['CountryA', 'CountryB', 'CountryC']

    for bank_id in range(n_banks):
        country = np.random.choice(countries)
        base_roa = np.random.uniform(0.5, 1.5)
        base_nim = np.random.uniform(2.0, 4.0)

        for year in years:
            # Add some random variation per year
            roa = base_roa + np.random.normal(0, 0.2)
            nim = base_nim + np.random.normal(0, 0.5)
            roe = roa * np.random.uniform(10, 15)
            eff_ratio = np.random.normal(60, 10)
            fee_share = np.random.normal(20, 5)

            # Deposits
            dep_demand = np.random.uniform(10, 30)
            dep_savings = np.random.uniform(20, 40)
            dep_time = np.random.uniform(20, 40)

            branches = np.random.randint(50, 500)
            dbranches_pct = np.random.normal(0, 5)

            data.append({
                'bank_id': bank_id,
                'year': year,
                'country': country,
                'NIM': nim,
                'ROA': roa,
                'ROE': roe,
                'Efficiency_Ratio': eff_ratio,
                'Fee_Income_Share': fee_share,
                'Dep_Demand_Share': dep_demand,
                'Dep_Savings_Share': dep_savings,
                'Dep_Time_Share': dep_time,
                'Branches': branches,
                'dBranches_pct': dbranches_pct
            })

    return pd.DataFrame(data)

def load_data():
    filepath = 'banks_reshaped.parquet'
    try:
        print(f"Attempting to load {filepath}...")
        df = pd.read_parquet(filepath)
        print("Data loaded successfully.")
        return df
    except Exception as e:
        print(f"Could not load {filepath}: {e}")
        print("Generating mock data...")
        return generate_mock_data()

if __name__ == "__main__":
    # Load Data
    DF = load_data()
    print(f"Data shape: {DF.shape}")
    print("Columns:", DF.columns.tolist())

    # ================== READY-TO-RUN TESTS (WITH SAFETY FILTERS) ==================
    # User provided code below

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
    # Check if we have columns before describe to avoid errors if all are missing
    cols_to_describe = [c for c in ["NIM","ROA","ROE","Efficiency_Ratio"] if c in DF2.columns]
    if cols_to_describe:
        print(DF2[cols_to_describe].describe(percentiles=[.01,.5,.99]).T[["count","mean","50%","1%","99%"]])
    else:
        print("No key variables found for description.")

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
    if PanelOLS is not None:
        try:
            reg_vars = ["ROA","NIM","Efficiency_Ratio","Fee_Income_Share"]
            missing_vars = [v for v in reg_vars if v not in DF2.columns]

            if not missing_vars:
                reg_df = DF2.dropna(subset=reg_vars).copy()
                reg_df = reg_df.set_index(["bank_id","year"])

                y = reg_df["ROA"]
                X = reg_df[["NIM","Efficiency_Ratio","Fee_Income_Share"]]
                X = sm.add_constant(X)

                mod = PanelOLS(y, X, entity_effects=True, time_effects=True)
                res = mod.fit(cov_type="clustered", cluster_entity=True)

                print("\n================ BASELINE PANELOLS (ROA) ================")
                print(res.summary)
            else:
                print(f"\nPanel regression skipped. Missing columns: {missing_vars}")

        except Exception as e:
            print("\nPanel regression skipped (error during execution):", str(e))
    else:
        print("\nPanel regression skipped (linearmodels not available).")
