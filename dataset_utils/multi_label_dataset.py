import os
import torch
import pandas as pd
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

# === 配置路径 ===
FEATURE_PATH = 'rawdata/nodes_all.xlsx'
EDGE_INDEX_PATH = 'rawdata/Adj.xlsx'
EDGE_ATTR_PATH = 'rawdata/edges_len.xlsx'
OUTPUT_DIR = 'dataset_utils/pyg_data_multi'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 读取数据 ===
nodes_df = pd.read_excel(FEATURE_PATH)
adj_df = pd.read_excel(EDGE_INDEX_PATH)
edge_len_df = pd.read_excel(EDGE_ATTR_PATH)

# === 清洗异常值 ===
numeric_cols = ['Elevation', 'DIS_AV_CMS', 'avg_prec', 'avg_tmax', 'avg_tmin', 'FRI', 'DRI']
nodes_df[numeric_cols] = nodes_df[numeric_cols].replace(-9999, pd.NA)

# === 处理流量缺失值 ===
def fill_flow_feature(df, col='DIS_AV_CMS'):
    for basin, group in df.groupby('RIVER_BASIN'):
        for idx, row in group.iterrows():
            if pd.isna(row[col]):
                upstream = adj_df[adj_df['NodeID'] == row['NodeID']]['EdgeID'].values
                flow_vals = group[col].dropna().values
                if len(flow_vals):
                    df.at[idx, col] = flow_vals.mean()
    return df

nodes_df = fill_flow_feature(nodes_df, 'DIS_AV_CMS')

# === 处理每个流域 ===
groups = nodes_df.groupby('RIVER_BASIN')
scaler = StandardScaler()
feature_cols = ['Elevation', 'DIS_AV_CMS', 'avg_prec', 'avg_tmax', 'avg_tmin']

for basin, group in groups:
    node_ids = group['NodeID'].tolist()
    features = group[feature_cols].copy()
    labels = group[['FRI', 'DRI']].copy()

    # 标准化特征并填充缺失
    features = pd.DataFrame(scaler.fit_transform(features.fillna(features.mean())), columns=feature_cols)
    labels = labels.fillna(torch.nan)

    x = torch.tensor(features.values, dtype=torch.float)
    y = torch.tensor(labels.values, dtype=torch.float)

    # 构造边
    edge_list = []
    edge_features = []
    for edge_id, rows in adj_df.groupby('EdgeID'):
        src = int(rows.iloc[0]['NodeID'])
        tgt = int(rows.iloc[1]['NodeID'])
        if src in node_ids and tgt in node_ids:
            edge_list.append([node_ids.index(src), node_ids.index(tgt)])
            length = edge_len_df[edge_len_df['EdgeID'] == edge_id]['Length'].values[0]
            edge_features.append([length])

    if len(edge_list) == 0:
        continue

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.basin_name = basin

    out_path = os.path.join(OUTPUT_DIR, f"{basin}.pt")
    torch.save(data, out_path)
    print(f"✅ Saved: {out_path}")

print("🎉 All multi-label graphs saved to:", OUTPUT_DIR)
