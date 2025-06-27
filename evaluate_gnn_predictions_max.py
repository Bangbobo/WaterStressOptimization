import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ndmo_red_node_max import ThreeLayerGIN, replace_pyg_flow_with_absolute, load_all_basins, replace_flow_and_evaluate_multitask

# 配置
data_path = "dataset_utils/pyg_data_multi"
raw_data_path = "rawdata/nodes_all.xlsx"
fri_model_path = "results/best_models/GIN_FRI_best_r2_0.5970.pth"
dri_model_path = "results/best_models/GIN_DRI_best_r2_0.6041.pth"
save_csv = "results/global_analysis/basin_prediction_errors_node_max.csv"
save_fig = "results/global_analysis/gnn_prediction_scatter_node_max.pdf"

os.makedirs(os.path.dirname(save_csv), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型
fri_model = ThreeLayerGIN(5, 256, 2).to(device)
fri_model.load_state_dict(torch.load(fri_model_path, map_location=device))
fri_model.eval()

dri_model = ThreeLayerGIN(5, 256, 2).to(device)
dri_model.load_state_dict(torch.load(dri_model_path, map_location=device))
dri_model.eval()

# 数据
df = pd.read_excel(raw_data_path)
basin_groups = df.groupby("RIVER_BASIN")
basin_files = [f for f in os.listdir(data_path) if f.endswith(".pt")]
all_basins = load_all_basins(raw_data_path)

records = []

for file in basin_files:
    basin_name = file.replace(".pt", "")
    if basin_name not in all_basins or basin_name not in basin_groups.groups:
        continue
    try:
        group = basin_groups.get_group(basin_name).sort_values(by="NodeID")
        true_fri_values = group["FRI"].values
        true_dri_values = group["DRI"].values

        flows = all_basins[basin_name]
        data = torch.load(os.path.join(data_path, file), map_location='cpu')
        data = replace_pyg_flow_with_absolute(data, flows)

        # 模型预测节点值
        _, _, flow_mean, flow_std = replace_flow_and_evaluate_multitask(data, flows, fri_model, dri_model)
        data.x[:, 1] = torch.tensor(flows, dtype=torch.float32)
        x_np = data.x.numpy()
        x_np = (x_np - x_np.mean(axis=0)) / (x_np.std(axis=0) + 1e-6)
        data.x = torch.tensor(x_np, dtype=torch.float32)

        with torch.no_grad():
            fri_pred = fri_model(data).cpu().numpy()[:, 0]
            dri_pred = dri_model(data).cpu().numpy()[:, 1]

        max_node_fri_idx = np.argmax(true_fri_values)
        max_node_dri_idx = np.argmax(true_dri_values)

        records.append({
            "Basin": basin_name,
            "FRI_node_max_index": max_node_fri_idx,
            "FRI_true": true_fri_values[max_node_fri_idx],
            "FRI_pred": fri_pred[max_node_fri_idx],
            "FRI_err": fri_pred[max_node_fri_idx] - true_fri_values[max_node_fri_idx],
            "DRI_node_max_index": max_node_dri_idx,
            "DRI_true": true_dri_values[max_node_dri_idx],
            "DRI_pred": dri_pred[max_node_dri_idx],
            "DRI_err": dri_pred[max_node_dri_idx] - true_dri_values[max_node_dri_idx]
        })
    except Exception as e:
        print(f"❌ Error in {basin_name}: {e}")

# 保存
out_df = pd.DataFrame(records)
out_df.to_csv(save_csv, index=False)

# 绘图
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].scatter(out_df["FRI_true"], out_df["FRI_pred"], color='teal', alpha=0.6)
axs[0].plot([out_df["FRI_true"].min(), out_df["FRI_true"].max()],
            [out_df["FRI_true"].min(), out_df["FRI_true"].max()],
            linestyle="--", color="black")
axs[0].set_xlabel("True Max Node FRI")
axs[0].set_ylabel("Pred Max Node FRI")
axs[0].set_title("Max Node FRI Prediction")

axs[1].scatter(out_df["DRI_true"], out_df["DRI_pred"], color='darkorange', alpha=0.6)
axs[1].plot([out_df["DRI_true"].min(), out_df["DRI_true"].max()],
            [out_df["DRI_true"].min(), out_df["DRI_true"].max()],
            linestyle="--", color="black")
axs[1].set_xlabel("True Max Node DRI")
axs[1].set_ylabel("Pred Max Node DRI")
axs[1].set_title("Max Node DRI Prediction")

plt.tight_layout()
plt.savefig(save_fig)
plt.close()