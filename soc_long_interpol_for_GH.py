import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit  # pip install scipy

# ---- 1) INPUTS ---------------------------------------------------------------
EXCEL_PATH = "enter path here"  # Excel file path

# Sheet name to load; if None, the first sheet is used
SHEET_NAME = None

YEARS_COL = "Years"
SCENARIO2_COL = "Scenario2"
BASELINE_COL = "Baseline"

N_BOOT = 1000
SEED = 42

# ---- 2) HELPERS --------------------------------------------------------------
def to_float_series(x: pd.Series) -> pd.Series:
    """Convert strings with commas and spaces to float."""
    if pd.api.types.is_numeric_dtype(x):
        return x.astype(float)
    return (
        x.astype(str)
         .str.replace("\u00A0", "", regex=False)
         .str.replace(" ", "", regex=False)
         .str.replace(",", ".", regex=False)
         .str.replace(r"[^0-9\.\-eE]", "", regex=True)
         .replace({"": np.nan})
         .astype(float)
    )

def logistic(t, K, r, t0):
    return K / (1.0 + np.exp(-r * (t - t0)))

def fit_logistic(t, y):
    K0 = (np.nanmax(y) * 1.1) if np.nanmax(y) > 0 else (np.nanmean(y) + 1.0)
    r0 = 0.1
    t0 = np.nanmedian(t)
    p0 = (K0, r0, t0)

    eps = 1e-6
    K_low = max(eps, np.nanmin(y) * 0.1) if np.nanmin(y) > 0 else eps
    K_high = max(10 * np.nanmax(y), eps + 1.0)
    tmin, tmax = float(np.nanmin(t)), float(np.nanmax(t))
    bounds = ([K_low, 1e-6, tmin - 20], [K_high, 5.0, tmax + 20])

    popt, pcov = curve_fit(logistic, t, y, p0=p0, bounds=bounds, maxfev=20000)
    return popt, pcov

def safe_percent(numer, denom):
    out = np.full_like(numer, np.nan, dtype=float)
    mask = np.isfinite(denom) & (np.abs(denom) > 0)
    out[mask] = (numer[mask] / denom[mask]) * 100.0
    return out

# ---- 3) LOAD ONE SHEET -------------------------------------------------------
xls = pd.ExcelFile(EXCEL_PATH)
sheet_to_use = SHEET_NAME if isinstance(SHEET_NAME, (str, int)) else xls.sheet_names[0]
df = xls.parse(sheet_to_use)

# ---- 4) COLUMN HANDLING ------------------------------------------------------
df.columns = [str(c).strip() for c in df.columns]
cols_lower = {c.lower(): c for c in df.columns}
ycol = cols_lower.get(YEARS_COL.lower(), YEARS_COL)
scol = cols_lower.get(SCENARIO2_COL.lower(), SCENARIO2_COL)
bcol = cols_lower.get(BASELINE_COL.lower(), BASELINE_COL)

for c in (ycol, scol, bcol):
    if c not in df.columns:
        raise ValueError(f"Column '{c}' not found. Available: {list(df.columns)}")

df = df[[ycol, scol, bcol]].copy().sort_values(ycol).reset_index(drop=True)

years = to_float_series(df[ycol]).to_numpy()
s2 = to_float_series(df[scol]).to_numpy()
baseline = to_float_series(df[bcol]).to_numpy()

# ---- 5) TIME INDEX -----------------------------------------------------------
t = np.arange(1, len(years) + 1, dtype=float)
if len(t) < 30:
    raise ValueError("At least 30 rows required for fitting indices 10–30.")

# ---- 6) LOGISTIC FIT (indices 10–30, Scenario 2) -----------------------------
fit_mask = (t >= 10) & (t <= 30)
t_fit = t[fit_mask]
y_fit = s2[fit_mask]

popt, _ = fit_logistic(t_fit, y_fit)
s2_fit = logistic(t, *popt)

# ---- 7) BOOTSTRAP CONFIDENCE INTERVALS --------------------------------------
rng = np.random.default_rng(SEED)
n_fit = len(t_fit)
last_30_start = int(t[-1] - 30 + 1)
last30_mask = t >= last_30_start

abs_means = []
pct_means = []
preds = []

for _ in range(N_BOOT):
    idx = rng.integers(0, n_fit, size=n_fit)
    tb, yb = t_fit[idx], y_fit[idx]
    if np.all(tb == tb[0]):
        continue
    try:
        popt_b, _ = fit_logistic(tb, yb)
    except Exception:
        continue
    s2_pred = logistic(t, *popt_b)
    preds.append(s2_pred)

    diff = s2_pred - baseline
    diff_pct = safe_percent(diff, baseline)
    abs_means.append(np.nanmean(diff[last30_mask]))
    pct_means.append(np.nanmean(diff_pct[last30_mask]))

if len(preds) == 0:
    raise RuntimeError("All bootstrap fits failed; check data in indices 10–30.")

preds = np.array(preds)
low = np.nanpercentile(preds, 2.5, axis=0)
high = np.nanpercentile(preds, 97.5, axis=0)
med = np.nanmedian(preds, axis=0)

# ---- 8) SUMMARY STATS (last 30 years) ---------------------------------------
diff_med = med - baseline
diff_pct_med = safe_percent(diff_med, baseline)

abs_point = float(np.nanmean(diff_med[last30_mask]))
pct_point = float(np.nanmean(diff_pct_med[last30_mask]))
abs_lo, abs_hi = np.nanpercentile(abs_means, [2.5, 97.5])
pct_lo, pct_hi = np.nanpercentile(pct_means, [2.5, 97.5])

print("\n=== Scenario2 vs Baseline — Last 30 Years ===")
print(f"Absolute increase (S2 - Baseline): {abs_point:,.4f}")
print(f"95% CI (absolute): [{abs_lo:,.4f}, {abs_hi:,.4f}]")
print(f"Percent increase: {pct_point:,.2f}%")
print(f"95% CI (percent): [{pct_lo:,.2f}%, {pct_hi:,.2f}%]")

# ---- 9) PLOTS ---------------------------------------------------------------
# A: Scenario 2 vs Baseline with 95% band
plt.figure(figsize=(10, 6))
plt.plot(years, s2, label="Scenario 2 (data)")
plt.plot(years, s2_fit, linestyle="--", label="Logistic fit (original)")
plt.fill_between(years, low, high, alpha=0.25, label="95% bootstrap band (S2)")
plt.plot(years, baseline, label="Baseline")
plt.axvspan(years[t >= last_30_start][0], years[t >= last_30_start][-1],
            alpha=0.1, label="Last 30 years")
plt.title("Scenario 2 vs Baseline — Logistic Fit + 95% Band")
plt.xlabel("Year")
plt.ylabel("SOC (units)")
plt.legend()
plt.tight_layout()

# B: Difference (S2 - Baseline) with band
diff_low = low - baseline
diff_high = high - baseline
diff_med = med - baseline

plt.figure(figsize=(10, 6))
plt.plot(years, diff_med, label="Median (S2 - Baseline)")
plt.fill_between(years, diff_low, diff_high, alpha=0.25, label="95% band of difference")
plt.axhline(0.0, linestyle=":")
plt.axvspan(years[t >= last_30_start][0], years[t >= last_30_start][-1],
            alpha=0.1, label="Last 30 years")
plt.title("Difference: Scenario 2 minus Baseline")
plt.xlabel("Year")
plt.ylabel("Difference (SOC units)")
plt.legend()
plt.tight_layout()

plt.show()
