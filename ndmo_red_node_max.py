# === 更新后的 ndmo_red.py ===
import torch
import torch.nn as nn
from torch_geometric.data import Data
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# === 通用 GNN 结构可复用 ===
class ThreeLayerGIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        from torch_geometric.nn import GINConv, BatchNorm
        from torch_geometric.nn.models import MLP

        self.conv1 = GINConv(MLP([input_dim, hidden_dim, hidden_dim]))
        self.bn1 = BatchNorm(hidden_dim)
        self.conv2 = GINConv(MLP([hidden_dim, hidden_dim, hidden_dim]))
        self.bn2 = BatchNorm(hidden_dim)
        self.conv3 = GINConv(MLP([hidden_dim, hidden_dim, output_dim]))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        return x

# === 替换 PyG 数据集中的流量特征 ===
def replace_pyg_flow_with_absolute(data, node_flows):
    data.x[:, 1] = torch.tensor(node_flows, dtype=torch.float32)
    return data

# === 加载所有盆地的节点流量 ===
def load_all_basins(raw_data_path):
    raw_data = pd.read_excel(raw_data_path)
    basin_data = {}
    for basin_name, group in raw_data.groupby("RIVER_BASIN"):
        node_flows = group.sort_values(by="NodeID")["DIS_AV_CMS"].values
        basin_data[basin_name] = node_flows
    return basin_data

# === 生成初始种群 ===
def generate_initial_population(data, steps=5, n_population=300, max_trials=5000):
    edge_index = data.edge_index.numpy()
    n_nodes = data.x.shape[0]
    xl = 0.5 * data.x[:, 1].numpy()
    xu = 1.5 * data.x[:, 1].numpy()

    candidate_flows = [np.linspace(xl[node], xu[node], steps) for node in range(n_nodes)]
    initial_population = np.array([np.array([candidate_flows[node][i] for node in range(n_nodes)]) for i in range(steps)])
    initial_population = np.unique(initial_population, axis=0)

    trial_count = 0
    while len(initial_population) < n_population:
        trial_count += 1
        if trial_count > max_trials:
            print("Warning: Max trials reached while generating initial population.")
            break
        parent1, parent2 = initial_population[np.random.choice(len(initial_population), size=2, replace=False)]
        child = crossover(parent1, parent2)
        mutated_child = mutate(child, mutation_rate=0.2, xl=xl, xu=xu)
        if check_feasibility_relaxed(mutated_child, edge_index, xl, xu):
            initial_population = np.vstack([initial_population, mutated_child])
            initial_population = np.unique(initial_population, axis=0)

    return initial_population[:n_population]

# === 交叉与变异 ===
def crossover(parent1, parent2):
    if len(parent1) <= 1 or len(parent2) <= 1:
        return parent1
    point = np.random.randint(1, len(parent1) - 1)
    return np.concatenate((parent1[:point], parent2[point:]))

def mutate(solution, mutation_rate=0.1, xl=None, xu=None):
    for i in range(len(solution)):
        if np.random.rand() < mutation_rate:
            solution[i] = np.random.uniform(xl[i], xu[i])
    return solution

# === 可行性约束 ===
def check_feasibility_relaxed(solution, edge_index, xl, xu, epsilon=1e-1, allow_violation_ratio=0.5):
    violations = 0
    for j, i in zip(*edge_index):
        if solution[i] + epsilon < solution[j]:  # 放松：下游不小于上游，可有微小误差
            violations += 1
    violation_ratio = violations / edge_index.shape[1]
    return violation_ratio <= allow_violation_ratio and np.all((solution >= xl) & (solution <= xu))

# === 替换流量并计算多目标任务预测 ===

def replace_flow_and_evaluate_multitask(data, flow_array, fri_model, dri_model):
    data.x[:, 1] = torch.tensor(flow_array, dtype=torch.float32)

    x_np = data.x.numpy()
    means = x_np.mean(axis=0)
    stds = x_np.std(axis=0) + 1e-6
    x_np = (x_np - means) / stds
    data.x = torch.tensor(x_np, dtype=torch.float32)

    with torch.no_grad():
        fri_out = fri_model(data).cpu().numpy()[:, 0]
        dri_out = dri_model(data).cpu().numpy()[:, 1]

    return np.max(fri_out), np.max(dri_out), means[1], stds[1]  # 返回流量特征的 mean 和 std


# === 快速非支配排序 ===
def fast_non_dominated_sort(objectives):
    num_solutions = len(objectives)
    domination_counts = np.zeros(num_solutions, dtype=int)
    dominated_solutions = [[] for _ in range(num_solutions)]
    fronts = [[]]

    for i in range(num_solutions):
        for j in range(num_solutions):
            if all(objectives[i] <= objectives[j]) and any(objectives[i] < objectives[j]):
                dominated_solutions[i].append(j)
            elif all(objectives[j] <= objectives[i]) and any(objectives[j] < objectives[i]):
                domination_counts[i] += 1

        if domination_counts[i] == 0:
            fronts[0].append(i)

    current_front = 0
    while len(fronts[current_front]) > 0:
        next_front = []
        for i in fronts[current_front]:
            for j in dominated_solutions[i]:
                domination_counts[j] -= 1
                if domination_counts[j] == 0:
                    next_front.append(j)
        fronts.append(next_front)
        current_front += 1

    return fronts[:-1]

# === 拥挤度计算 ===
def compute_crowding_distance(objectives):
    num_points, num_objectives = objectives.shape
    distance = np.zeros(num_points)
    for i in range(num_objectives):
        sorted_idx = np.argsort(objectives[:, i])
        distance[sorted_idx[0]] = distance[sorted_idx[-1]] = np.inf
        min_val, max_val = objectives[sorted_idx[0], i], objectives[sorted_idx[-1], i]
        if max_val - min_val == 0:
            continue
        for j in range(1, num_points - 1):
            prev = objectives[sorted_idx[j - 1], i]
            next = objectives[sorted_idx[j + 1], i]
            distance[sorted_idx[j]] += (next - prev) / (max_val - min_val)
    return distance

# === 可视化与保存结果 ===
def plot_population_and_pareto_front_with_original(all_F, pareto_F, original_F, basin_name, save_dir):
    plt.figure(figsize=(10, 6))
    plt.scatter(all_F[:, 0], all_F[:, 1], color='green', alpha=0.6, label='All Solutions')
    plt.scatter(pareto_F[:, 0], pareto_F[:, 1], color='blue', alpha=0.9, label='Pareto Front')
    plt.scatter(original_F[0], original_F[1], color='red', s=80, label='Original Flow', zorder=5)
    plt.xlabel("FRI", fontsize=14)
    plt.ylabel("DRI", fontsize=14)
    plt.title(f"Pareto Front for {basin_name}", fontsize=16)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    save_path = os.path.join(save_dir, f"{basin_name}_pareto_front_with_original.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches="tight")
    plt.close()
    print(f"✅ Saved Pareto front plot for {basin_name} → {save_path}")

def save_pareto_solutions_with_scores(solutions, fri_scores, dri_scores,
                                      save_dir, basin_name,
                                      original_flow=None,
                                      original_fri=None,
                                      original_dri=None):
    save_path = os.path.join(save_dir, f"{basin_name}_pareto_solutions_with_scores.csv")
    hv = fri_scores * dri_scores

    df_rows = []
    row_labels = []
    # 如果提供了原始流，则添加第一行（历史流）
    if original_flow is not None:
        orig_hv = original_fri * original_dri
        row_labels.append("Historical flow")
        # 将原始流（列表形式）、原始 FRI、原始 DRI 和计算得到的 HV 拼接成一行数据
        row = list(original_flow) + [original_fri, original_dri, orig_hv]
        df_rows.append(row)

    # 添加每个 Pareto 解，行索引依次为 Pareto solution 1, Pareto solution 2, ...
    for i, (sol, f1, f2, h) in enumerate(zip(solutions, fri_scores, dri_scores, hv)):
        row_labels.append(f"Pareto solution {i + 1}")
        row = list(sol) + [f1, f2, h]
        df_rows.append(row)

    # 定义列名称：节点信息 + 两个评价指标和 HV 列
    col_names = [f"Node_{i}" for i in range(solutions.shape[1])] + ["ParetoBestFRI", "ParetoBestDRI", "HV"]
    # 创建 DataFrame，并用 row_labels 作为行索引
    df = pd.DataFrame(df_rows, columns=col_names, index=row_labels)

    # 保存 CSV 时保留行索引
    df.to_csv(save_path, index=True)
    print(f"✅ Saved Pareto solutions → {save_path}")