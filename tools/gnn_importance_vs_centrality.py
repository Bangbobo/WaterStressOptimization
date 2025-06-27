# ==== tools/gnn_importance_vs_centrality.py ====
import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from tqdm import tqdm

def analyze_importance_vs_centrality(label="DRI"):
    importance_dir = f"results/importance/{label.lower()}"
    importance_files = [f for f in os.listdir(importance_dir) if f.endswith("_importance.npy")]

    records = []

    for f in tqdm(importance_files, desc=f"Analyzing {label}"):
        basin_name = f.replace("_importance.npy", "")
        try:
            data = np.load(os.path.join(importance_dir, f), allow_pickle=True).item()
            node_score = data["node_importance"]
            edge_score = data["edge_importance"]
            edge_index = data["edge_index"]

            G = nx.Graph()
            for i in range(edge_index.shape[1]):
                u, v = edge_index[:, i]
                G.add_edge(int(u), int(v))
            if G.number_of_nodes() < 2:
                continue

            degree_dict = dict(G.degree())
            betweenness_dict = nx.betweenness_centrality(G, normalized=True)
            edge_betweenness_dict = nx.edge_betweenness_centrality(G, normalized=True)

            degree = np.array([degree_dict.get(i, 0) for i in range(len(node_score))])
            betweenness = np.array([betweenness_dict.get(i, 0) for i in range(len(node_score))])
            edge_betweenness = np.array([edge_betweenness_dict.get((int(u), int(v)), 0) for u, v in zip(*edge_index)])

            if len(degree) < 2:
                continue

            pearson_d = np.corrcoef(degree, node_score)[0, 1]
            spearman_d = pd.Series(degree).corr(pd.Series(node_score), method='spearman')
            pearson_b = np.corrcoef(betweenness, node_score)[0, 1]
            spearman_b = pd.Series(betweenness).corr(pd.Series(node_score), method='spearman')
            if len(edge_score) > 2:
                pearson_e = np.corrcoef(edge_betweenness, edge_score)[0, 1]
                spearman_e = pd.Series(edge_betweenness).corr(pd.Series(edge_score), method='spearman')
            else:
                pearson_e, spearman_e = np.nan, np.nan

            records.append({
                "Basin": basin_name,
                "Pearson_Degree": pearson_d,
                "Spearman_Degree": spearman_d,
                "Pearson_Betweenness": pearson_b,
                "Spearman_Betweenness": spearman_b,
                "Pearson_EdgeBetweenness": pearson_e,
                "Spearman_EdgeBetweenness": spearman_e,
            })

        except Exception as e:
            print(f"⚠️ Error in {f}: {e}")

    df_corr = pd.DataFrame(records)
    os.makedirs("results/global_analysis", exist_ok=True)
    df_corr.to_csv(f"results/global_analysis/importance_centrality_{label.lower()}.csv", index=False)

if __name__ == "__main__":
    #analyze_importance_vs_centrality(label="FRI")
    analyze_importance_vs_centrality(label="DRI")
