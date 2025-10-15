
"""
SOC storage trajectories — smoothed vs raw.
Generates moving-average columns and plots smoothed vs original series.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ------------------------------------------------------------------
# 1) File paths (edit placeholders)
# ------------------------------------------------------------------
input_file  = Path("enter path here")   # Excel input
output_file = Path("enter path here")   # Excel with *_MA columns
figure_file = Path("enter path here")   # Output figure (PNG)

# ------------------------------------------------------------------
# 2) Load data and build moving-average columns
# ------------------------------------------------------------------
df = pd.read_excel(input_file)

columns_to_process = ["Scenario1", "Scenario2", "Scenario3", "Baseline"]
colors  = ["darkgreen", "lightblue", "salmon", "brown"]
labels  = ["Scenario 1", "Scenario 2", "Scenario 3", "Baseline"]

window_size = 15  # moving average window
for col in columns_to_process:
    df[f"{col}_MA"] = df[col].rolling(window=window_size, min_periods=1).mean()

# Optional: save dataframe with added MA columns
df.to_excel(output_file, index=False)
print(f"Smoothed data saved: {output_file}")

# ------------------------------------------------------------------
# 3) Plot — convert kg → Mg (kg / 1,000)
# ------------------------------------------------------------------
SCALE = 1_000  # kg → Mg

fig, ax = plt.subplots(figsize=(10, 6))

for col, color, label in zip(columns_to_process, colors, labels):
    smoothed = df[f"{col}_MA"] / SCALE
    original = df[col] / SCALE

    ax.plot(smoothed, color=color, linewidth=3, label=label)
    ax.plot(original, linestyle="--", alpha=0.3, color=color, label=f"{label} Original")

# Legend: smoothed first, originals second
handles, lbls = ax.get_legend_handles_labels()
ax.legend(
    handles[::2] + handles[1::2],
    lbls[::2] + lbls[1::2],
    fontsize=11, ncol=2, loc="lower right", columnspacing=1.5
)

# ------------------------------------------------------------------
# 4) Axis formatting
# ------------------------------------------------------------------
BASE = 67.7  # vertical offset added to tick labels (display only)
ax.yaxis.set_major_locator(mticker.MultipleLocator(2))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{int(round(y + BASE))}"))

ax.set_title("SOC storage increase for each Scenario [Mg C ha⁻¹]", fontsize=14)
ax.set_xlabel("Years", fontsize=12)
ax.set_ylabel("SOC storage [Mg C ha⁻¹]", fontsize=12)
ax.grid(True)

fig.tight_layout()

# ------------------------------------------------------------------
# 5) Save figure
# ------------------------------------------------------------------
plt.savefig(figure_file, dpi=600)
plt.show()
print(f"Figure saved: {figure_file}")
