import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import gaussian_kde

# 加载数据
df = pd.read_csv("results/PF_relaxed_max/summary_report.csv")
df["FRI_surplus"] = pd.to_numeric(df["OrigFRI"], errors="coerce") - pd.to_numeric(df["ParetoBestFRI"], errors="coerce")
df["DRI_surplus"] = pd.to_numeric(df["OrigDRI"], errors="coerce") - pd.to_numeric(df["ParetoBestDRI"], errors="coerce")

df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["FRI_surplus", "DRI_surplus"])
df_clean["FRI_surplus"] = df_clean["FRI_surplus"].astype(float)
df_clean["DRI_surplus"] = df_clean["DRI_surplus"].astype(float)

# KDE 数据
fri = df_clean["FRI_surplus"].values
dri = df_clean["DRI_surplus"].values
fri_kde = gaussian_kde(fri)
dri_kde = gaussian_kde(dri)
fri_range = np.linspace(fri.min(), fri.max(), 500)
dri_range = np.linspace(dri.min(), dri.max(), 500)

# 创建双图布局
fig = plt.figure(figsize=(14, 5))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.4, 1])

# Panel (a): Scatter
ax0 = plt.subplot(gs[0])
sc = ax0.scatter(
    df_clean["FRI_surplus"], df_clean["DRI_surplus"],
    c=df_clean["NumPareto"], cmap="plasma", alpha=0.8, linewidth=0.3
)
ax0.set_xlabel("FRI Surplus (Orig - Optimal)", fontsize=13,fontweight="bold")
ax0.set_ylabel("DRI Surplus (Orig - Optimal)", fontsize=13,fontweight="bold")
ax0.set_title("(a) Tradeoff between DRI and FRI surplus", fontsize=13,loc="center",fontweight="bold")
cb = plt.colorbar(sc, ax=ax0)
cb.set_label("Number of Pareto Solutions", fontsize=13,loc="center",fontweight="bold")
cb.ax.tick_params(labelsize=10)
ax0.tick_params(axis='both', labelsize=11)
ax0.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

# Panel (b): Density plots
ax1 = plt.subplot(gs[1])
ax1.plot(fri_range, fri_kde(fri_range), label="FRI Surplus", color="teal", linewidth=2)
ax1.fill_between(fri_range, fri_kde(fri_range), color="teal", alpha=0.3)

ax1.plot(dri_range, dri_kde(dri_range), label="DRI Surplus", color="darkorange", linewidth=2)
ax1.fill_between(dri_range, dri_kde(dri_range), color="darkorange", alpha=0.3)

ax1.set_xlabel("Surplus (Orig - Optimal)", fontsize=13,fontweight="bold")
ax1.set_ylabel("Density", fontsize=13,fontweight="bold")
ax1.set_title("(b) Marginal distribution of surplus", fontsize=13,loc="center",fontweight="bold")
ax1.legend(fontsize=13)
ax1.tick_params(axis='both', labelsize=11)
ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

plt.tight_layout()
plt.savefig("results/global_analysis/surplus_joint_figure.pdf", dpi=300)
plt.show()
