import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score

# === 路径配置 ===
DATA_DIR = 'dataset_utils/pyg_data_multi'
SPLIT_DIR = 'dataset_utils/split_masks/multi'
os.makedirs(SPLIT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(42)
torch.manual_seed(42)

# === 对抗性模型 ===
class AdversarialModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

# === 训练与评估 ===
def train_adversarial(model, data_loader, optimizer, criterion):
    model.train()
    for data in data_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()

def evaluate_adversarial(model, data_loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            preds.append(model(data.x).cpu())
            targets.append(data.y.cpu())
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    return roc_auc_score(targets, preds)

# === 创建掩码 ===
def random_split_nodes(data, ratio=0.8):
    n = data.num_nodes
    perm = torch.randperm(n)
    split = int(n * ratio)
    train_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[perm[:split]] = True
    test_mask[perm[split:]] = True
    return train_mask, test_mask

def save_split_to_txt(train_mask, test_mask, filename, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    train_nodes = torch.where(train_mask)[0].cpu().numpy()
    test_nodes = torch.where(test_mask)[0].cpu().numpy()
    with open(os.path.join(save_dir, filename + '.txt'), 'w') as f:
        f.write("Train Nodes:\n")
        f.write(", ".join(map(str, train_nodes)))
        f.write("\nTest Nodes:\n")
        f.write(", ".join(map(str, test_nodes)))

# === 主逻辑 ===
all_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.pt')])
valid_auc_low, valid_auc_high = 0.45, 0.55

for i, f in enumerate(all_files):
    path = os.path.join(DATA_DIR, f)
    data = torch.load(path)
    success = False
    for attempt in range(10):
        train_mask, test_mask = random_split_nodes(data)
        y = torch.zeros(data.num_nodes)
        y[test_mask] = 1
        data.y = y.unsqueeze(-1)
        data.train_mask = train_mask
        data.test_mask = test_mask

        model = AdversarialModel(input_dim=data.x.size(1)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCELoss()

        loader = DataLoader([data], batch_size=1)
        for _ in range(10):
            train_adversarial(model, loader, optimizer, criterion)
        auc = evaluate_adversarial(model, loader)
        print(f"[{i+1}/{len(all_files)}] {f} | AUC={auc:.3f}")

        if valid_auc_low <= auc <= valid_auc_high:
            save_split_to_txt(train_mask, test_mask, f"multi_split_{f.replace('.pt', '')}", SPLIT_DIR)
            success = True
            break

    if not success:
        print(f"⚠️ {f} failed to generate valid split after 10 attempts.")

print("✅ All valid splits saved to:", SPLIT_DIR)