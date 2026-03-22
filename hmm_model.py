"""
Hidden Markov Model for market regime detection.


Fits a 3-state Gaussian HMM to FRED macroeconomic data and outputs a
D3-ready JSON file (regime_output.json) with:
  - Per-month regime labels and posterior probabilities
  - Transition probability matrix
  - Per-regime summary statistics

Dependencies:
    pip install hmmlearn scikit-learn pandas numpy

Usage:
    In CMD run: 
        python hmm_model.py
        python -m http.server 8000
        Open your browser type in: http://localhost:8000/
        Navigate to the 'Project' folder
"""

import json
import warnings
import os
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# Configuration 

DATA_PATH   = os.path.join(os.path.dirname(__file__), "FRED_DATA.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "regime_output.json")

FEATURE_COLS = ["SP500_return", "yield_spread", "UNRATE", "CPI_change"]

N_REGIMES    = 3
N_RESTARTS   = 20     # HMM has better performance around 20 for this, keep testing when we add more
N_ITER       = 200
FFILL_LIMIT  = 3      # max consecutive months to forward-fill sparse columns
RANDOM_SEED  = 42

# Output regime ordering and display metadata (Growth=0, Crisis=1, Transition=2)
REGIME_META = [
    {"id": 0, "key": "growth",     "label": "Low-Vol Growth",        "color": "#2ecc71"},
    {"id": 1, "key": "crisis",     "label": "High-Vol Crisis",        "color": "#e74c3c"},
    {"id": 2, "key": "transition", "label": "Transition/Tightening",  "color": "#f39c12"},
]

# JSON encoder

class NumpyEncoder(json.JSONEncoder):
    # This handles numpy scalar and array types, and converts NaN to null.
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _safe(v):
    # This converts numpy scalar / Python float NaN - None for JSON serialization
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, np.floating):
        return None if np.isnan(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


# Step 1: Load & Rename

def load_data(filepath: str) -> pd.DataFrame:
    # Load FRED CSV, parse dates, coerce numerics, rename columns.
    df = pd.read_csv(filepath)

    # All non-Date columns to numeric (blank strings to NaN)
    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df = df.set_index("Date").sort_index()

    df = df.rename(columns={
        "USREC_Recession_Indicator": "USREC",
        "UNRATE_Unemployment_Rate":  "UNRATE",
        "CPIAUCSL_CPI":              "CPI",
        "DGS10_10Y_Treasury":        "DGS10",
    })

    return df


# Step 2 & 3: Preprocess + Feature Engineering

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Forward-fill sparse columns (no backfill - We want to avoid look-ahead bias)
    df["SP500"] = df["SP500"].ffill(limit=FFILL_LIMIT)
    df["DGS10"] = df["DGS10"].ffill(limit=FFILL_LIMIT)

    ################ Input Features here ###########################
    df["SP500_return"] = df["SP500"].pct_change() # monthly equity - sidenote, pd is forward-filling pct, insert fill_method this line and below
    df["CPI_change"]   = df["CPI"].pct_change()   # monthly inflation rate
    df["yield_spread"] = df["DGS10"] - df["FEDFUNDS"]   # yield curve proxy

    # Not using 10Y Treasury Yield in HMM
    # Monthly return should be = to coupon income + price change from modified duration
    # price_change approx -modified_duration × (Delta)yield  (modified duration aprox 9 for 10Y par bond)
    BOND_DURATION = 9.0
    df["bond_return"] = (
        df["DGS10"].shift(1) / 1200
        - BOND_DURATION * df["DGS10"].diff() / 100
    )

    return df


# Step 4: Scale Features

def scale_features(df: pd.DataFrame, feature_cols: list):
    # Drop rows with NaN in feature columns, fit StandardScaler, return scaled array.
    df_model = df.dropna(subset=feature_cols).copy()
    X_raw    = df_model[feature_cols].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    return X_scaled, scaler, df_model


# Step 5: Train HMM (multiple restarts)

def fit_hmm_best(X_scaled: np.ndarray, n_components: int = N_REGIMES,
                 n_restarts: int = N_RESTARTS, n_iter: int = N_ITER,
                 seed_base: int = RANDOM_SEED):
    # Fit GaussianHMM with multiple random restarts; return model with highest log-likelihood.
    best_model = None
    best_score = -np.inf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # put the warning ignore here
        for i in range(n_restarts):
            model = GaussianHMM(
                n_components=n_components,
                covariance_type="full", # Swtiched from full
                n_iter=n_iter,
                tol=1e-4,
                random_state=seed_base + i,
                min_covar = 1e-3, # Check this if we want to keep it.
            )
            try:
                model.fit(X_scaled)
                score = model.score(X_scaled)
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception:
                continue   # degenerate covariance on this seed, skip

    if best_model is None:
        raise RuntimeError("All HMM restarts failed: Check data quality.")

    return best_model, best_score


# Step 6: Label Regimes

def label_regimes(model: GaussianHMM, scaler: StandardScaler,
                  feature_cols: list) -> dict:
    """
    Map arbitrary internal state indices.

    Third pass strategy (using state means in original units):
      1. Lowest SP500_return:  'crisis'
      2. Highest SP500_return: 'growth'
      3. Remainder:            'transition'
    """
    means_orig = scaler.inverse_transform(model.means_)  # (n_states, n_features)
    sp500_idx  = feature_cols.index("SP500_return")
    unrate_idx = feature_cols.index("UNRATE")

    # Composite score: higher = more "growth-like"
    scores = means_orig[:, sp500_idx] - 0.3 * means_orig[:, unrate_idx]
    order  = np.argsort(scores)   # ascending: [crisis_id, transition_id, growth_id]

    state_label_map = {
        int(order[0]): "crisis",
        int(order[1]): "transition",
        int(order[2]): "growth",
    }
    return state_label_map


# Step 7: Decode States

def decode_states(model: GaussianHMM, X_scaled: np.ndarray):
    # Return Viterbi state and probabilities smoothing.
    hidden_states = model.predict(X_scaled)          # shape (n,)
    state_probs   = model.predict_proba(X_scaled)    # shape (n, 3) - Refine as needed
    return hidden_states, state_probs


# Step 8: Diagnostics check for development - Not needed for viz

def print_diagnostics(model: GaussianHMM, df_model: pd.DataFrame,
                      hidden_states: np.ndarray, state_label_map: dict,
                      scaler: StandardScaler, feature_cols: list,
                      log_likelihood: float):
    n = len(df_model)
    n_params = (N_REGIMES ** 2 - N_REGIMES) + (N_REGIMES - 1) \
               + N_REGIMES * len(feature_cols) \
               + N_REGIMES * len(feature_cols) ** 2
    aic = 2 * n_params - 2 * log_likelihood
    bic = n_params * np.log(n) - 2 * log_likelihood

    print("\nHMM TRAINING DIAGNOSTICS:")
    print(f"Features       : {feature_cols}")
    print(f"Observations   : {n} rows  "
          f"({df_model.index[0].date()} - {df_model.index[-1].date()})")
    print(f"Log-likelihood : {log_likelihood:.2f}")
    print(f"n_params       : {n_params}")
    print(f"AIC            : {aic:.2f}")
    print(f"BIC            : {bic:.2f}")

    # Regime distribution
    print("\n Regime Distribution:")
    key_to_label = {r["key"]: r["label"] for r in REGIME_META}
    for state_id, key in sorted(state_label_map.items()):
        count = int((hidden_states == state_id).sum())
        print(f"  {key_to_label[key]:<28}  {count:>3} months  ({100*count/n:.1f}%)")

    # Transition matrix (original state order, with labels)
    print("\n Transition Matrix (row = from, col = to)")
    labels = [key_to_label[state_label_map[i]] for i in range(N_REGIMES)]
    col_w  = max(len(l) for l in labels) + 2
    header = " " * col_w + "".join(f"{l:>{col_w}}" for l in labels)
    print("  " + header)
    for i in range(N_REGIMES):
        row = f"  {labels[i]:<{col_w}}" + "".join(
            f"{model.transmat_[i, j]:>{col_w}.3f}" for j in range(N_REGIMES))
        print(row)

    # State means in original units
    means_orig = scaler.inverse_transform(model.means_)
    print("\n State Means (original units)")
    header2 = f"  {'State':<28}" + "".join(f"{c:>15}" for c in feature_cols)
    print(header2)
    for state_id in range(N_REGIMES):
        key   = state_label_map[state_id]
        label = key_to_label[key]
        row   = f"  {label:<28}" + "".join(
            f"{means_orig[state_id, j]:>15.4f}" for j in range(len(feature_cols)))
        print(row)

    # Recession overlap sanity check
    if "USREC" in df_model.columns:
        crisis_id = next(sid for sid, k in state_label_map.items() if k == "crisis")
        rec_months   = df_model["USREC"] == 1
        crisis_months = hidden_states == crisis_id
        if rec_months.sum() > 0:
            overlap = (rec_months.values & crisis_months).sum()
            pct     = 100 * overlap / rec_months.sum()
            print("\n Recession Overlap (validation)")
            print(f"  USREC=1 months captured by High-Vol Crisis: "
                  f"{overlap}/{int(rec_months.sum())} ({pct:.0f}%)")


# Max drawdown helper

def _max_drawdown(returns: pd.Series) -> float:
    # Max peak-to-trough drawdown from a series of period returns. Returns a negative fraction, e.g. -0.35 means -35% drawdown.
    valid = returns.dropna()
    if len(valid) < 2:
        return 0.0
    cumret = (1 + valid).cumprod()
    running_max = cumret.cummax()
    dd = (cumret - running_max) / running_max
    return float(dd.min())


# Compute state statistics

def compute_state_statistics(df_model: pd.DataFrame,
                              hidden_states: np.ndarray,
                              state_label_map: dict) -> list:
    # Return per-regime summary statistics over the modeled time series.
    n_total    = len(df_model)
    key_to_id  = {r["key"]: r["id"] for r in REGIME_META}
    key_to_lbl = {r["key"]: r["label"] for r in REGIME_META}
    stats_list = []

    for state_id, key in state_label_map.items():
        mask   = hidden_states == state_id
        subset = df_model.loc[mask]
        n      = int(mask.sum())

        rec_overlap = float(
            (subset["USREC"] == 1).sum() / n if n > 0 and "USREC" in subset.columns
            else 0.0
        )

        # Equity-bond correlation (only when enough bond data is present)
        bond_data = subset["bond_return"] if "bond_return" in subset.columns else pd.Series(dtype=float)
        eq_bond_corr = (
            _safe(subset["SP500_return"].corr(bond_data))
            if bond_data.notna().sum() > 5
            else None
        )

        stats_list.append({
            "regime_id":            key_to_id[key],
            "label":                key_to_lbl[key],
            "n_months":             n,
            "pct_months":           round(n / n_total, 4),
            "recession_overlap_pct": round(rec_overlap, 4),
            "mean_SP500_return":    _safe(subset["SP500_return"].mean()),
            "std_SP500_return":     _safe(subset["SP500_return"].std()),
            "mean_bond_return":     _safe(bond_data.mean()),
            "std_bond_return":      _safe(bond_data.std()),
            "max_drawdown_equity":  _safe(_max_drawdown(subset["SP500_return"])),
            "max_drawdown_bond":    _safe(_max_drawdown(bond_data)),
            "equity_bond_corr":     eq_bond_corr,
            "mean_FEDFUNDS":        _safe(subset["FEDFUNDS"].mean()),
            "mean_DGS10":           _safe(subset["DGS10"].mean()),
            "mean_UNRATE":          _safe(subset["UNRATE"].mean()),
            "mean_CPI_change":      _safe(subset["CPI_change"].mean()),
            "mean_yield_spread":    _safe(subset["yield_spread"].mean()),
        })

    # Sort by regime output id
    stats_list.sort(key=lambda x: x["regime_id"])
    return stats_list


# Build transition matrix (reordered to output regime IDs) 

def _build_transition_matrix(model: GaussianHMM, state_label_map: dict) -> list:
    # Return 3×3 transition matrix reordered to match output regime IDs (0=Growth, 1=Crisis, 2=Transition).
    key_to_id  = {r["key"]: r["id"] for r in REGIME_META}
    # Map: output_id → internal_state_id
    out_id_to_state = {key_to_id[key]: sid
                       for sid, key in state_label_map.items()}
    order = [out_id_to_state[i] for i in range(N_REGIMES)]   # [growth_sid, crisis_sid, trans_sid]

    matrix = []
    for from_out in range(N_REGIMES):
        row = []
        for to_out in range(N_REGIMES):
            row.append(round(float(model.transmat_[order[from_out], order[to_out]]), 6))
        matrix.append(row)
    return matrix


# Step 9: This is building output JSON

def build_output_json(df: pd.DataFrame, df_model: pd.DataFrame,
                      hidden_states: np.ndarray, state_probs: np.ndarray,
                      model: GaussianHMM, state_label_map: dict,
                      scaler: StandardScaler, feature_cols: list,
                      log_likelihood: float) -> dict:
    # Assemble the output dictionary.
    # All rows from the original df are included in time_series
    # Rows not used in the model have regime = null
    key_to_id  = {r["key"]: r["id"] for r in REGIME_META}
    key_to_lbl = {r["key"]: r["label"] for r in REGIME_META}

    # Model metadata
    n = len(df_model)
    n_params = (N_REGIMES ** 2 - N_REGIMES) + (N_REGIMES - 1) \
               + N_REGIMES * len(feature_cols) \
               + N_REGIMES * len(feature_cols) ** 2
    aic = 2 * n_params - 2 * log_likelihood
    bic = n_params * np.log(n) - 2 * log_likelihood

    # Build a lookup: date -> (regime_out_id, regime_label, probs_by_out_id)
    # probs are reordered to match output regime IDs [growth, crisis, transition]
    key_to_id = {r["key"]: r["id"] for r in REGIME_META}
    out_id_to_state = {key_to_id[key]: sid for sid, key in state_label_map.items()}
    # prob columns in output id order
    prob_col_order = [out_id_to_state[i] for i in range(N_REGIMES)]

    regime_by_date = {}
    for i, (idx, _) in enumerate(df_model.iterrows()):
        internal_id  = int(hidden_states[i])
        key          = state_label_map[internal_id]
        out_id       = key_to_id[key]
        probs        = [round(float(state_probs[i, col]), 6) for col in prob_col_order]
        regime_by_date[idx] = {
            "regime":       out_id,
            "regime_label": key_to_lbl[key],
            "regime_probs": probs,
        }

    # Build time series in original
    raw_cols = ["FEDFUNDS", "SP500", "UNRATE", "CPI", "DGS10", "USREC"]
    derived  = ["SP500_return", "bond_return", "CPI_change", "yield_spread"]
    time_series = []
    for date, row in df.iterrows():
        entry = {"date": date.strftime("%Y-%m-%d")}

        if date in regime_by_date:
            entry.update(regime_by_date[date])
        else:
            entry.update({"regime": None, "regime_label": None, "regime_probs": None})

        for col in raw_cols:
            entry[col] = _safe(row.get(col, None))

        for col in derived:
            entry[col] = _safe(row.get(col, None))

        time_series.append(entry)

    return {
        "metadata": {
            "n_regimes":        N_REGIMES,
            "features_used":    feature_cols,
            "date_range": {
                "start": df.index[0].strftime("%Y-%m-%d"),
                "end":   df.index[-1].strftime("%Y-%m-%d"),
            },
            "model_date_range": {
                "start": df_model.index[0].strftime("%Y-%m-%d"),
                "end":   df_model.index[-1].strftime("%Y-%m-%d"),
            },
            "n_observations":   n,
            "log_likelihood":   round(log_likelihood, 4),
            "aic":              round(aic, 4),
            "bic":              round(bic, 4),
        },
        "regimes":           REGIME_META,
        "transition_matrix": _build_transition_matrix(model, state_label_map),
        "time_series":       time_series,
        "state_statistics":  compute_state_statistics(df_model, hidden_states,
                                                      state_label_map),
    }


# JSON file for viz
def save_json(output_dict: dict, path: str):
    with open(path, "w") as f:
        json.dump(output_dict, f, cls=NumpyEncoder, indent=2)
    size_kb = os.path.getsize(path) / 1024
    print(f"Saved: {path}  ({size_kb:.1f} KB)")



if __name__ == "__main__":
    print(f"Loading data from:  {DATA_PATH}")
    df_raw = load_data(DATA_PATH)
    print(f"Loaded {len(df_raw)} rows  "
          f"({df_raw.index[0].date()} -> {df_raw.index[-1].date()})")

    df_clean = preprocess(df_raw)

    print(f"Scaling features:   {FEATURE_COLS}")
    X_scaled, scaler, df_model = scale_features(df_clean, FEATURE_COLS)
    print(f"Usable rows:        {len(df_model)}  "
          f"({df_model.index[0].date()} -> {df_model.index[-1].date()})")

    print(f"Training HMM ({N_RESTARTS} restarts, {N_ITER} EM iterations) …")
    model, log_likelihood = fit_hmm_best(X_scaled)

    state_label_map = label_regimes(model, scaler, FEATURE_COLS)
    hidden_states, state_probs = decode_states(model, X_scaled)

    print_diagnostics(
        model, df_model, hidden_states, state_label_map,
        scaler, FEATURE_COLS, log_likelihood
    )

    print("Building output JSON …")
    output = build_output_json(
        df_clean, df_model, hidden_states, state_probs,
        model, state_label_map, scaler, FEATURE_COLS, log_likelihood,
    )

    save_json(output, OUTPUT_PATH)
    print("Done.")