# ==== tools/explainer_gin_summary.py ====
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# tools/explainer_gnn.py
# 通用 GNN 模型解释器（支持 GCN/GAT/GIN 多任务）

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer
from torch_geometric.explain.config import ModelConfig
from GNNs.models import (
    ThreeLayerGCNWithEdgeFeatures,
    ThreeLayerGATWithEdgeFeatures,
    ThreeLayerGIN,
    ThreeLayerGraphTransformer
)


def load_best_model(model_type, label_name, input_dim, edge_feature_dim, device):
    model_dir = "results/best_models"
    model_files = [f for f in os.listdir(model_dir) if label_name in f and model_type in f and f.endswith(".pth")]
    if not model_files:
        raise FileNotFoundError(f"No model found for {model_type}_{label_name}")
    model_file = sorted(model_files, key=lambda x: float(x.split("r2_")[-1].replace(".pth", "")), reverse=True)[0]
    print(f"✅ Using model: {model_file}")

    model_path = os.path.join(model_dir, model_file)
    if model_type == 'GCN':
        model = ThreeLayerGCNWithEdgeFeatures(input_dim, 256, 2).to(device)
    elif model_type == 'GAT':
        model = ThreeLayerGATWithEdgeFeatures(input_dim, 256, 2, edge_feature_dim).to(device)
    elif model_type == 'GIN':
        model = ThreeLayerGIN(input_dim, 256, 2).to(device)
    elif model_type == 'GT':
        model = ThreeLayerGraphTransformer(input_dim, 256, 2, edge_feature_dim).to(device)
    else:
        raise ValueError("Unsupported model type")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


class ModelWrapper(torch.nn.Module):
    def __init__(self, model, target_index):
        super().__init__()
        self.model = model
        self.target_index = target_index

    def forward(self, x, edge_index, **kwargs):
        out = self.model(Data(x=x, edge_index=edge_index, edge_attr=kwargs.get("edge_attr")))
        return out[:, self.target_index]  # 选择目标维度


def explain_single_basin(model, data, device, basin_name, label_name):
    data = data.to(device)
    target_index = 0 if label_name.upper() == 'DRI' else 1

    explainer = Explainer(
        model=ModelWrapper(model, target_index),
        algorithm=GNNExplainer(epochs=200),
        explanation_type="model",
        model_config=ModelConfig(mode="regression", task_level="node", return_type="raw"),
        edge_mask_type='object'
    )

    explanation = explainer(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, target=data.y[:, target_index])
    edge_mask = explanation.edge_mask.cpu().numpy()
    edge_mask_norm = (edge_mask - edge_mask.min()) / (edge_mask.max() - edge_mask.min() + 1e-6)

    G = to_networkx(data, to_undirected=False)
    node_score = np.zeros(data.x.shape[0])
    for (u, v), score in zip(G.edges(), edge_mask_norm):
        node_score[u] += score * 0.8
        node_score[v] += score * 0.2
    node_score = (node_score - node_score.min()) / (node_score.max() - node_score.min() + 1e-6)

    result = {
        "node_importance": node_score,
        "edge_importance": edge_mask_norm,
        "edge_index": data.edge_index.cpu().numpy()
    }

    out_dir = f"results/importance/{label_name.lower()}"
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{basin_name}_importance.npy"), result)
    print(f"📌 Importance saved: {basin_name}_importance.npy")

    return result


def run_batch_summary(data_dir, model_type, label_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = [f for f in os.listdir(data_dir) if f.endswith(".pt")]
    sample_data = torch.load(os.path.join(data_dir, files[0]))
    input_dim = sample_data.x.shape[1]
    edge_feature_dim = sample_data.edge_attr.shape[1]

    model = load_best_model(model_type, label_name, input_dim, edge_feature_dim, device)
    for file in files:
        basin_name = file.replace(".pt", "")
        print(f"🧠 Explaining {basin_name}...")
        data = torch.load(os.path.join(data_dir, file))
        explain_single_basin(model, data, device, basin_name, label_name)


if __name__ == "__main__":
    run_batch_summary("dataset_utils/pyg_data_multi", model_type="GIN", label_name="DRI")
