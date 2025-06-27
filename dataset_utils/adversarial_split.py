# ==== data_utils/adversarial_split.py ====

import os
import torch
import random
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

# ==== 配置 ====
LABELS = ['FRI', 'DRI']
INPUT_DIM = 5
INPUT_DIR_BASE = 'dataset_utils/pyg_data_{}'
SAVE_MASK_DIR_BASE = 'dataset_utils/split_masks/{}'

os.makedirs('dataset_utils/split_masks/fri', exist_ok=True)
os.makedirs('dataset_utils/split_masks/dri', exist_ok=True)

# ==== 对抗性验证模型 ====
class AdversarialModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AdversarialModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

def train_adversarial(model, loader, optimizer, criterion, device):
    model.train()
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()

def evaluate_adversarial(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data.x)
            preds.append(output.cpu())
            labels.append(data.y.cpu())
    return roc_auc_score(torch.cat(labels), torch.cat(preds))

def save_mask_to_txt(train_mask, test_mask, filename):
    train_nodes = torch.where(train_mask)[0].cpu().numpy()
    test_nodes = torch.where(test_mask)[0].cpu().numpy()
    with open(filename, 'w') as f:
        f.write("Train Nodes:\n")
        f.write(", ".join(map(str, train_nodes)))
        f.write("\nTest Nodes:\n")
        f.write(", ".join(map(str, test_nodes)))

# ==== 执行划分 ====
def adversarial_split(label_name):
    input_dir = INPUT_DIR_BASE.format(label_name.lower())
    save_dir = SAVE_MASK_DIR_BASE.format(label_name.lower())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    failed_list = []

    for file in sorted(os.listdir(input_dir)):
        if not file.endswith('.pt'):
            continue
        path = os.path.join(input_dir, file)
        data = torch.load(path)
        num_nodes = data.num_nodes

        max_iter = 10
        for attempt in range(max_iter):
            # 随机划分节点
            indices = torch.randperm(num_nodes)
            split_idx = int(0.8 * num_nodes)
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            train_mask[indices[:split_idx]] = True
            test_mask[indices[split_idx:]] = True

            # 构造新的标签
            data.train_mask = train_mask
            data.test_mask = test_mask
            data.y = torch.zeros(num_nodes)
            data.y[test_mask] = 1
            data.y = data.y.unsqueeze(-1)

            loader = DataLoader([data], batch_size=1)
            model = AdversarialModel(INPUT_DIM, 64).to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.BCELoss()

            for _ in range(10):
                train_adversarial(model, loader, optimizer, criterion, device)

            auc = evaluate_adversarial(model, loader, device)
            print(f"{file} | Attempt {attempt + 1} | AUC: {auc:.4f}")

            if 0.45 <= auc <= 0.55:
                basin_name = file.replace('.pt', '')
                save_path = os.path.join(save_dir, f"{label_name.lower()}_split_{basin_name}.txt")
                save_mask_to_txt(train_mask, test_mask, save_path)
                print(f"✅ Saved mask for {basin_name}")
                break
        else:
            print(f"❌ Failed to find valid split for {file} after {max_iter} attempts")
            failed_list.append(file)

    # 保存失败列表
    fail_log_path = os.path.join(save_dir, f"{label_name.lower()}_split_failures.txt")
    with open(fail_log_path, 'w') as f:
        for name in failed_list:
            f.write(name + '\n')
    print(f"📄 Saved failure list to {fail_log_path}")

if __name__ == '__main__':
    for label in LABELS:
        print(f"\n=== Processing {label} datasets ===")
        adversarial_split(label)
    print("\n✅ All done!")