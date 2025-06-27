import os
import torch
from torch_geometric.data import Data
import numpy as np

# ==== 配置路径 ====
RAW_DATA_DIR_FRI = 'dataset_utils/pyg_data_fri'
RAW_DATA_DIR_DRI = 'dataset_utils/pyg_data_dri'
SAVE_PATH = 'dataset_utils/pyg_data_multi'
os.makedirs(SAVE_PATH, exist_ok=True)


# ==== 图级特征构造函数 ====
def compute_meta_features(data):
    x = data.x

    # 替代 torch.nanmean 和 nanstd
    x_no_nan = torch.nan_to_num(x, nan=0.0)
    mask = ~torch.isnan(x)
    count = mask.sum(dim=0).clamp(min=1)

    mean_features = (x_no_nan * mask).sum(dim=0) / count
    std_features = torch.sqrt(((x_no_nan - mean_features) ** 2 * mask).sum(dim=0) / count)

    max_features = torch.nan_to_num(torch.max(x, dim=0).values, nan=0.0)
    min_features = torch.nan_to_num(torch.min(x, dim=0).values, nan=0.0)

    num_nodes = torch.tensor([x.size(0)], dtype=torch.float)
    num_edges = torch.tensor([data.edge_index.size(1)], dtype=torch.float)
    avg_degree = num_edges / num_nodes

    meta = torch.cat([
        mean_features, std_features, max_features, min_features,
        num_nodes, num_edges, avg_degree
    ])
    return meta

# ==== 主转换函数 ====
file_list = sorted([f for f in os.listdir(RAW_DATA_DIR_FRI) if f.endswith('.pt')])

for file in file_list:
    path_fri = os.path.join(RAW_DATA_DIR_FRI, file)
    path_dri = os.path.join(RAW_DATA_DIR_DRI, file)

    if not os.path.exists(path_fri) or not os.path.exists(path_dri):
        print(f"Skipping missing pair: {file}")
        continue

    data_fri = torch.load(path_fri)
    data_dri = torch.load(path_dri)

    assert torch.allclose(data_fri.x, data_dri.x), f"Mismatch in x for {file}"
    assert torch.equal(data_fri.edge_index, data_dri.edge_index), f"Mismatch in edge_index for {file}"
    assert torch.allclose(data_fri.edge_attr, data_dri.edge_attr), f"Mismatch in edge_attr for {file}"

    # 构造 multi-label y
    y_fri = data_fri.y.unsqueeze(-1)
    y_dri = data_dri.y.unsqueeze(-1)
    y_multi = torch.cat([y_fri, y_dri], dim=1)

    data_fri.y = y_multi
    data_fri.meta = compute_meta_features(data_fri)  # 添加图级特征

    save_path = os.path.join(SAVE_PATH, file)
    torch.save(data_fri, save_path)
    print(f"✅ Saved multi-label graph to {save_path}")
