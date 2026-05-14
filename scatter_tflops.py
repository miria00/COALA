"""
scatter_tflops.py  —  Win Rate vs TFLOPS from COALA paper (Table 1 + Table 3)
Run:  python scatter_tflops.py
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Table 3: TFLOPS on EduFeedback dataset ────────────────────────────────────
TFLOPS = {
    "Mistral-7B": {"COALA": 1580.45, "DPO": 9284.71,  "ORPO": 11241.89, "SFT": 2492},
    "Dolphin-7B": {"COALA": 1794.66, "DPO": 10091.25, "ORPO": 12116.50, "SFT": 2804},
    "LLaMA-8B":   {"COALA": 1805.39, "DPO": 10253.37, "ORPO": 12352.98, "SFT": 2851},
}

# ── Table 1: AlpacaEval2 LC Win Rate % (Edu dataset) ─────────────────────────
LC_WR_EDU = {
    "Mistral-7B": {"COALA": 24.61, "DPO": 24.19, "ORPO": 17.01, "SFT": 6.80},
    "Dolphin-7B": {"COALA": 40.81, "DPO": 34.73, "ORPO": 25.06, "SFT": 17.36},
    "LLaMA-8B":   {"COALA": 40.90, "DPO": 40.68, "ORPO": 23.87, "SFT": 10.92},
}

COLORS  = {"COALA": "#1D9E75", "DPO": "#378ADD", "ORPO": "#E24B4A", "SFT": "#888780"}
MARKERS = {"Mistral-7B": "o", "Dolphin-7B": "^", "LLaMA-8B": "s"}
MARKER_SIZE = 120

fig, ax = plt.subplots(figsize=(8, 5.5))

# Light background fill for COALA region
coala_tflops = [TFLOPS[m]["COALA"] for m in TFLOPS]
ax.axvspan(min(coala_tflops) * 0.8, max(coala_tflops) * 1.2,
           alpha=0.06, color="#1D9E75", zorder=0)

for model, methods in TFLOPS.items():
    for method, tflops in methods.items():
        wr = LC_WR_EDU[model][method]
        ax.scatter(tflops, wr,
                   color=COLORS[method],
                   marker=MARKERS[model],
                   s=MARKER_SIZE, zorder=4,
                   edgecolors="white", linewidths=0.8)

# Connect same model across methods with thin lines
for model in TFLOPS:
    points = [(TFLOPS[model][m], LC_WR_EDU[model][m]) for m in ["SFT", "ORPO", "DPO", "COALA"]]
    xs, ys = zip(*points)
    ax.plot(xs, ys, color="grey", linewidth=0.6, alpha=0.3, zorder=2, linestyle="--")

ax.set_xscale("log")
ax.set_xlabel("TFLOPS (log scale)", fontsize=13)
ax.set_ylabel("AlpacaEval2 LC Win Rate %", fontsize=13)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, alpha=0.15, which="both")
ax.tick_params(labelsize=11)

# ── Both legends in lower right, side by side ────────────────────────────────
# Method legend (colors)
method_handles = [
    plt.scatter([], [], color=c, s=80, label=m, edgecolors="white", linewidths=0.7, marker="o")
    for m, c in COLORS.items()
]
# Model legend (shapes)
model_handles = [
    plt.scatter([], [], marker=mk, color="#555555", s=80, label=mo, edgecolors="white", linewidths=0.7)
    for mo, mk in MARKERS.items()
]

leg1 = ax.legend(handles=method_handles, title="Method",
                 loc="lower right", fontsize=9, title_fontsize=9.5,
                 framealpha=0.92, borderpad=0.6,
                 bbox_to_anchor=(1.0, 0.0))
ax.add_artist(leg1)

leg2 = ax.legend(handles=model_handles, title="Model",
                 loc="lower right", fontsize=9, title_fontsize=9.5,
                 framealpha=0.92, borderpad=0.6,
                 bbox_to_anchor=(0.72, 0.0))

plt.tight_layout()
plt.savefig("winrate_vs_tflops.png", dpi=200, bbox_inches="tight")
print("Saved -> winrate_vs_tflops.png")
