# amazon_yellow_flow_compare.py（对齐标题 + 色带优化 + 添加比例尺和指北针）

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.patches import FancyArrow
from matplotlib.colors import CenteredNorm
# 创建输出目录
os.makedirs("results/global_analysis", exist_ok=True)
norm = CenteredNorm(vcenter=0)
# 盆地定义
basins = {
    "Amazon": {
        "basin": "rawdata/AMAZON_basin.shp",
        "node": "rawdata/AMAZON_node.shp",
        "edge": "rawdata/AMAZON_edge.shp",
        "flow": "results/PF_relaxed_max/AMAZON (also AMAZONAS)_pareto_solutions_with_scores.csv",
        "label": "(a)",
    },
    "Yellow": {
        "basin": "rawdata/YELLOW_basin.shp",
        "node": "rawdata/YELLOW_node.shp",
        "edge": "rawdata/YELLOW_edge.shp",
        "flow": "results/PF_relaxed_max/YELLOW RIVER_pareto_solutions_with_scores.csv",
        "label": "(b)",
    },
}

fig, axes = plt.subplots(1, 2, figsize=(20, 10), constrained_layout=True, gridspec_kw={'width_ratios': [1, 1]})

for i, (name, info) in enumerate(basins.items()):
    basin = gpd.read_file(info["basin"]).to_crs(epsg=3857)
    nodes = gpd.read_file(info["node"]).to_crs(epsg=3857)
    edges = gpd.read_file(info["edge"]).to_crs(epsg=3857)

    df = pd.read_csv(info["flow"], index_col=0)
    flow_cols = [col for col in df.columns if col.startswith("Node_")]
    historical = df.loc["Historical flow", flow_cols].values.astype(float)
    pareto1 = df.loc["Pareto solution 1", flow_cols].values.astype(float)
    delta_flow = pareto1 - historical

    nodes = nodes.reset_index(drop=True)
    if len(delta_flow) != len(nodes):
        raise ValueError(f"Length mismatch in {name}: flow={len(delta_flow)}, nodes={len(nodes)}")
    nodes["flow_diff"] = delta_flow
    nodes["abs_diff"] = np.abs(delta_flow)
    nodes = nodes.sort_values("abs_diff")

    v = np.max(np.abs(nodes["flow_diff"]))
    norm = TwoSlopeNorm(vmin=-v, vcenter=0, vmax=v)

    ax = axes[i]
    basin.plot(ax=ax, facecolor="white", edgecolor="black", linewidth=1)
    edges.plot(ax=ax, edgecolor="gray", linewidth=0.6, alpha=0.4)
    # nodes.plot(
    #     ax=ax,
    #     column="flow_diff",
    #     cmap="seismic",  # 改为全光谱色带
    #     markersize=20,
    #     zorder=3,
    #     norm=norm
    # )
    # 添加透明度列（线性归一化到 0.1 - 1.0 区间，避免完全透明）
    nodes["alpha"] = 0.1 + 0.9 * (nodes["abs_diff"] - nodes["abs_diff"].min()) / (
                nodes["abs_diff"].max() - nodes["abs_diff"].min())

    # 手动绘制点，支持设置 alpha
    for _, row in nodes.iterrows():
        ax.scatter(
            row.geometry.x,
            row.geometry.y,
            color=plt.cm.seismic(norm(row.flow_diff)),
            s=80,
            alpha=row.alpha,
            zorder=3,
            edgecolors='none',  # 添加黑色边框辅助对比
        )

    # 图标题与编号
    ax.text(0.02, 0.98, info["label"], transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
    ax.set_title(f"{name} Basin", fontsize=14, pad=20)
    ax.add_artist(ScaleBar(
    dx=1,
    units='m',
    location='lower left',
    scale_loc='bottom',
    fixed_value=500_000,
    fixed_units='m',
    length_fraction=0.2,
    box_alpha=0,
    color='black',
    scale_formatter=lambda value, unit: f'{int(value / 1000)} km'))
    ax.set_axis_off()

    # # 添加独立色条，位置右下但完全可见
    # cax = inset_axes(ax, width="3%", height="30%", loc="lower left", borderpad=1)
    sm = plt.cm.ScalarMappable(cmap="seismic", norm=norm)
    # sm._A = []
    # fig.colorbar(sm, cax=cax, label="Flow Change (m³/s)", orientation="vertical", ticks=[-v, 0, v],format="%.0e" ) # ✅ 改为科学计数法)
# 创建右侧色条（非 inset）：
    cbar = fig.colorbar(
    sm,
    ax=ax,
    orientation="vertical",
    shrink=0.6,
    pad=0.02,  # 与图之间的间距
    format="%.0e",
    label="Flow Change (m³/s)"
    )
plt.savefig("results/global_analysis/amazon_yellow_flow_compare_sorted.pdf")
print("✅ 保存图像：results/global_analysis/amazon_yellow_flow_compare_sorted.pdf")