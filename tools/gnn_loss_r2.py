# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import numpy as np
# from matplotlib.colors import to_rgba
#
# # ========== 设置路径与模型颜色 ==========
# log_dir = "D:/Pycharm/waterStressFinal/results/gnn_train_log"
# model_color_map = {
#     "GIN": "#7552A7",
#     "GAT": "#F38B2F",
#     "GCN": "#088C00",
#     "GT": "#214EA7"
# }
#
# def extract_model_name(filename):
#     for m in model_color_map:
#         if m in filename:
#             return m
#     return "Other"
#
# log_files = [f for f in os.listdir(log_dir) if f.endswith(".csv")]
# log_data = {}
# model_metrics = []
#
# # ========== 加载数据并按模型分类 ==========
# for f in log_files:
#     try:
#         df = pd.read_csv(os.path.join(log_dir, f))
#         df["Epoch"] = df["Epoch"].astype(int)
#         df["avg_R2"] = (df["R2_FRI"] + df["R2_DRI"]) / 2
#         df_filtered = df[(df["Epoch"] >= 15) & (df["Epoch"] <= 100)]
#         model = extract_model_name(f)
#
#         model_metrics.append({
#             "Model": model,
#             "Config": f.replace("train_log_", "").replace(".csv", ""),
#             "Best avg R²": round(df_filtered["avg_R2"].max(), 3),
#             "Best R² (DRI)": round(df_filtered["R2_DRI"].max(), 3),
#             "Best R² (FRI)": round(df_filtered["R2_FRI"].max(), 3),
#             "Min Loss": round(df_filtered["Loss"].min(), 3),
#         })
#
#         log_data[f] = (df_filtered, model)
#     except Exception as e:
#         print(f"❌ Failed to load {f}: {e}")
#
# # ========== 按模型分组并绘图 ==========
# grouped_data = {}
# for f, (df, model) in log_data.items():
#     grouped_data.setdefault(model, []).append((f, df, df["avg_R2"].max()))
#
# fig, axs = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
#
# for model, entries in grouped_data.items():
#     entries.sort(key=lambda x: x[2])  # 按 avg_R2 排序
#     base_color = model_color_map[model]
#     rgba = to_rgba(base_color)
#     best_entry = entries[-1][0]  # 最优实验文件名
#
#     for f, df_plot, score in entries:
#         is_best = f == best_entry
#         alpha = 1.0 if is_best else 0.3
#         lw = 2.5 if is_best else 1.2
#         label = model if is_best else None
#
#         axs[0].plot(df_plot["Epoch"], df_plot["Loss"], color=rgba[:3]+(alpha,), linewidth=lw, label=label)
#         axs[1].plot(df_plot["Epoch"], df_plot["R2_DRI"], color=rgba[:3]+(alpha,), linewidth=lw)
#         axs[2].plot(df_plot["Epoch"], df_plot["R2_FRI"], color=rgba[:3]+(alpha,), linewidth=lw)
#
# # 设置标题与格式
# titles = ["(a) Loss Curve", r"(b) $R^2$ for DRI", r"(c) $R^2$ for FRI"]
# for ax, title in zip(axs, titles):
#     ax.set_title(title, fontsize=13)
#     ax.set_xlabel("Epoch", fontsize=12)
#     ax.set_ylabel("Value", fontsize=11)
#     ax.grid(True)
#     ax.tick_params(axis='x', labelrotation=0)
#
# # 图例
# axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=3, title="Model", fontsize=9, title_fontsize=10)
#
# plt.tight_layout()
# plt.savefig("D:/Pycharm/waterStressFinal/results/gnn_training_comparison_final.pdf", dpi=300)
# plt.close()
#
# # ========== 表格 ==========
# df_metrics = pd.DataFrame(model_metrics).sort_values(["Model", "Best avg R²"], ascending=[True, False])
# # --- 修复 Config 列下划线的 LaTeX 兼容性 ---
# df_metrics["Config"] = df_metrics["Config"].apply(lambda x: f"${x.replace('_', '\\_')}$")
#
# latex_table = df_metrics.to_latex(
#     index=False,
#     caption="Performance comparison of GNN models (GIN, GAT, GCN, GT) across different configurations. The best result within each model is highlighted in the training plots.",
#     label="tab:gnn_model_performance",
#     column_format="llrrrr",
#     float_format="%.3f",
#     escape=False  # 关键：允许 LaTeX 代码输出
# )
#
# print(latex_table)
# df_metrics.to_csv("D:/Pycharm/waterStressFinal/results/model_r2_table_final.csv", index=False)
#
import pandas as pd
import matplotlib.pyplot as plt

# === 设置路径和样式 ===
log_file = "D:/Pycharm/waterStressFinal/results/gnn_train_log/train_log_GIN_hd256_lr0.0005.csv"
model_label = "GIN-hd256-lr0.0005"
model_color = "#7552A7"

# === 加载训练日志数据 ===
df = pd.read_csv(log_file)
df["Epoch"] = df["Epoch"].astype(int)

# === 创建图形窗口（水平排列） ===
fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharex=True)

# --- 图 1: Loss ---
axs[0].plot(df["Epoch"], df["Loss"], color=model_color, linewidth=2, label=model_label)
axs[0].set_title("(a) Loss Curve", fontsize=13)
axs[0].set_xlabel("Epoch", fontsize=11)
axs[0].set_ylabel("Loss", fontsize=11)
axs[0].grid(True)
axs[0].legend(loc="upper right", fontsize=9)

# --- 图 2: R2_DRI ---
axs[1].plot(df["Epoch"], df["R2_DRI"], color=model_color, linewidth=2)
axs[1].set_title(r"(b) $R^2$ for DRI", fontsize=13)
axs[1].set_xlabel("Epoch", fontsize=11)
axs[1].set_ylabel(r"$R^2$", fontsize=11)
axs[1].grid(True)
axs[1].legend([model_label], loc="lower right", fontsize=9)

# --- 图 3: R2_FRI ---
axs[2].plot(df["Epoch"], df["R2_FRI"], color=model_color, linewidth=2)
axs[2].set_title(r"(c) $R^2$ for FRI", fontsize=13)
axs[2].set_xlabel("Epoch", fontsize=11)
axs[2].set_ylabel(r"$R^2$", fontsize=11)
axs[2].grid(True)
axs[2].legend([model_label], loc="lower right", fontsize=9)

# === 调整布局，保存图像 ===
plt.tight_layout(pad=2.0)
plt.savefig("D:/Pycharm/waterStressFinal/results/gin_best_model_training_curve.pdf", dpi=300)
plt.close()
