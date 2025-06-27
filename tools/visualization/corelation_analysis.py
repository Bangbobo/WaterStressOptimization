import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import os

# 创建输出目录
os.makedirs("results/global_analysis", exist_ok=True)
# 加载 shapefile
basin = gpd.read_file("rawdata/AMAZON_basin.shp")
nodes = gpd.read_file("rawdata/AMAZON_node.shp")
edges = gpd.read_file("rawdata/AMAZON_edge.shp")

# 投影统一为 EPSG:3857
basin = basin.to_crs(epsg=3857)
nodes = nodes.to_crs(epsg=3857)
edges = edges.to_crs(epsg=3857)

# 构建图结构并计算中心性
G = nx.Graph()
for _, row in edges.iterrows():
    coords = list(row.geometry.coords)
    if len(coords) >= 2:
        u = coords[0]
        v = coords[-1]
        G.add_edge(tuple(u), tuple(v))

node_bet = nx.betweenness_centrality(G, normalized=True)
edge_bet = nx.edge_betweenness_centrality(G, normalized=True)

nodes["betweenness"] = nodes.geometry.apply(lambda p: node_bet.get((p.x, p.y), 0))
edges["betweenness"] = edges.geometry.apply(
    lambda line: edge_bet.get((tuple(line.coords[0]), tuple(line.coords[-1])), 0)
)

# 设置统一中心性筛选阈值
node_threshold = np.percentile(nodes["betweenness"], 80)
edge_threshold = np.percentile(edges["betweenness"], 80)
nodes_selected = nodes[nodes["betweenness"] >= node_threshold].copy()
edges_selected = edges[edges["betweenness"] >= edge_threshold].copy()

# 提前设定统一 colorbar 上限（根据调试建议）
NODE_IMP_VMAX = 0.6
EDGE_IMP_VMAX = 0.6
NODE_BET_VMAX = 0.10
EDGE_BET_VMAX = 0.10

# # 定义绘图函数（重要性地图）
# def plot_importance_map(npy_path, out_pdf, label_text):
#     data = np.load(npy_path, allow_pickle=True).item()
#     node_imp = np.array(data["node_importance"])
#     edge_imp = np.array(data["edge_importance"])
#
#     nodes_selected["importance"] = [node_imp[i] if i < len(node_imp) else 0 for i in nodes_selected["NodeID"]]
#     edges_selected["importance"] = edge_imp[:len(edges_selected)]
#
#     fig, ax = plt.subplots(figsize=(10, 10))
#     basin.plot(ax=ax, facecolor="white", edgecolor="black", linewidth=1)
#     nodes_selected.plot(ax=ax, column="importance", cmap="YlOrRd", markersize=20, legend=True,
#                         legend_kwds={'label': "Node Importance", 'shrink': 0.6}, zorder=3, vmin=0, vmax=NODE_IMP_VMAX)
#     edges_selected.plot(ax=ax, column="importance", cmap="YlOrRd", linewidth=1.5, legend=True,
#                         legend_kwds={'label': "Edge Importance", 'shrink': 0.6}, vmin=0, vmax=EDGE_IMP_VMAX)
#
#     ax.text(0.02, 0.98, label_text, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
#     ax.set_axis_off()
#     plt.tight_layout()
#     plt.savefig(out_pdf)
#     plt.close()
#
# # 图 (a) 和 (b)：FRI 和 DRI 重要性图
# plot_importance_map("results/importance/fri/AMAZON (also AMAZONAS)_importance.npy", "results/global_analysis/amazon_map_fri_importance_detailed.pdf", "(a)")
# plot_importance_map("results/importance/dri/AMAZON (also AMAZONAS)_importance.npy", "results/global_analysis/amazon_map_dri_importance_detailed.pdf", "(b)")

# # 图 (c)：结构中心性图
# fig, ax = plt.subplots(figsize=(10, 10))
# basin.plot(ax=ax, facecolor="white", edgecolor="black", linewidth=1)
# nodes_selected.plot(ax=ax, column="betweenness", cmap="Blues", markersize=20, legend=True,
#                     legend_kwds={'label': "Node Betweenness", 'shrink': 0.6}, zorder=3, vmin=0, vmax=NODE_BET_VMAX)
# edges_selected.plot(ax=ax, column="betweenness", cmap="Blues", linewidth=1.5, legend=True,
#                     legend_kwds={'label': "Edge Betweenness", 'shrink': 0.6}, vmin=0, vmax=EDGE_BET_VMAX)
#
# ax.text(0.02, 0.98, "(c)", transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
# ax.set_axis_off()
# plt.tight_layout()
# plt.savefig("results/global_analysis/amazon_map_betweenness_detailed.pdf")
# plt.close()

# 图 (a) (b) (c) (d)：节点和边散点图（统一格式）
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import geopandas as gpd

# 加载数据
nodes = gpd.read_file("rawdata/AMAZON_node.shp")
edges = gpd.read_file("rawdata/AMAZON_edge.shp")

fri_data = np.load("results/importance/fri/AMAZON (also AMAZONAS)_importance.npy", allow_pickle=True).item()
dri_data = np.load("results/importance/dri/AMAZON (also AMAZONAS)_importance.npy", allow_pickle=True).item()
node_fri = np.array(fri_data["node_importance"])
node_dri = np.array(dri_data["node_importance"])
edge_fri = np.array(fri_data["edge_importance"])
edge_dri = np.array(dri_data["edge_importance"])

nodes["importance_fri"] = [node_fri[i] if i < len(node_fri) else 0 for i in nodes["NodeID"]]
nodes["importance_dri"] = [node_dri[i] if i < len(node_dri) else 0 for i in nodes["NodeID"]]
edges["importance_fri"] = edge_fri[:len(edges)]
edges["importance_dri"] = edge_dri[:len(edges)]

# betweenness
G = nx.Graph()
for _, row in edges.iterrows():
    coords = list(row.geometry.coords)
    if len(coords) >= 2:
        u = coords[0]
        v = coords[-1]
        G.add_edge(tuple(u), tuple(v))

node_bet = nx.betweenness_centrality(G, normalized=True)
edge_bet = nx.edge_betweenness_centrality(G, normalized=True)

nodes["betweenness"] = nodes.geometry.apply(lambda p: node_bet.get((p.x, p.y), 0))
edges["betweenness"] = edges.geometry.apply(lambda line: edge_bet.get((tuple(line.coords[0]), tuple(line.coords[-1])), 0))

# 绘图
fig = plt.figure(figsize=(12, 10))
gs = gridspec.GridSpec(2, 2)

plt.rcParams["font.family"] = "Arial"

def plot_scatter(ax, x, y, label, xlabel, ylabel, color):
    ax.scatter(x, y, s=10, alpha=0.5, color=color)
    ax.set_xlabel(xlabel, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.95, label, transform=ax.transAxes, fontsize=18, fontweight="bold", va="top")

plot_scatter(fig.add_subplot(gs[0, 0]),
             nodes["betweenness"], nodes["importance_fri"],
             "(a)", "Betweenness", "FRI Node Importance", "teal")

plot_scatter(fig.add_subplot(gs[0, 1]),
             nodes["betweenness"], nodes["importance_dri"],
             "(b)", "Betweenness", "DRI Node Importance", "darkorange")

plot_scatter(fig.add_subplot(gs[1, 0]),
             edges["betweenness"], edges["importance_fri"],
             "(c)", "Betweenness", "FRI Edge Importance", "darkviolet")

plot_scatter(fig.add_subplot(gs[1, 1]),
             edges["betweenness"], edges["importance_dri"],
             "(d)", "Betweenness", "DRI Edge Importance", "forestgreen")

plt.tight_layout()
plt.savefig("results/global_analysis/amazon_importance_scatter_defg.pdf")
plt.close()
