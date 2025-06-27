# ==== train_model.py ====
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import csv
import glob
from torch.utils.tensorboard import SummaryWriter
from GNNs.models import (
    ThreeLayerGCNWithEdgeFeatures,
    ThreeLayerGATWithEdgeFeatures,
    ThreeLayerGIN,
    ThreeLayerGraphTransformer
)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ==== 配置参数 ====
LABEL_NAME = 'multi'  # 'FRI', 'DRI', or 'multi' for both
MODEL_TYPES = ['GT'] #['GCN', 'GAT', 'GIN', 'GT']
INPUT_DIM = 5
EDGE_FEATURE_DIM = 1
OUTPUT_DIM = 2  # 2 for multi-task (FRI, DRI)
NUM_EPOCHS = 100
PATIENCE = 10
SAVE_ROOT = 'results'
SAVE_BEST_MODEL_DIR = os.path.join(SAVE_ROOT, 'best_models')
BATCH_SIZE = 1
HIDDEN_DIMS = [128, 256, 512]  # 👈 多隐藏维度组合
LEARNING_RATES = [1e-3, 5e-4, 1e-4]  # 👈 多学习率组合

os.makedirs(SAVE_ROOT, exist_ok=True)
os.makedirs(SAVE_BEST_MODEL_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Running on device: {device}\n")

# ==== EarlyStopping 类 ====
class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def step(self, score):
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# ==== 主训练逻辑 ====
for model_type in MODEL_TYPES:
    for hidden_dim in HIDDEN_DIMS:
        for lr in LEARNING_RATES:

            print(f"\n🔧 Training {model_type} | dim={hidden_dim} | lr={lr} for multi-task [FRI, DRI]")

            DATA_DIR = 'dataset_utils/pyg_data_multi'
            SPLIT_DIR = 'dataset_utils/split_masks/multi'
            LOG_NAME = f'{model_type}_hd{hidden_dim}_lr{lr}'
            CURVE_PNG = f'{SAVE_ROOT}/loss_curve_{LOG_NAME}.png'
            LOG_CSV = f'{SAVE_ROOT}/train_log_{LOG_NAME}.csv'
            LOG_TBOARD = f'{SAVE_ROOT}/tensorboard/{LOG_NAME}'

            writer = SummaryWriter(LOG_TBOARD)

            if model_type == 'GCN':
                model = ThreeLayerGCNWithEdgeFeatures(INPUT_DIM, hidden_dim, OUTPUT_DIM).to(device)
            elif model_type == 'GAT':
                model = ThreeLayerGATWithEdgeFeatures(INPUT_DIM, hidden_dim, OUTPUT_DIM, EDGE_FEATURE_DIM).to(device)
            elif model_type == 'GIN':
                model = ThreeLayerGIN(INPUT_DIM, hidden_dim, OUTPUT_DIM).to(device)
            elif model_type == 'GT':
                model = ThreeLayerGraphTransformer(INPUT_DIM, hidden_dim, OUTPUT_DIM, EDGE_FEATURE_DIM).to(device)
            else:
                raise ValueError("Unknown model type")

            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()
            early_stopper = EarlyStopping(PATIENCE)
            train_losses = []

            best_r2_fri = float('-inf')
            best_r2_dri = float('-inf')

            all_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.pt')])

            with open(LOG_CSV, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['Epoch', 'Loss', 'R2_FRI', 'R2_DRI'])

                for epoch in range(NUM_EPOCHS):
                    model.train()
                    total_loss = 0.0
                    all_fri_preds, all_fri_labels = [], []
                    all_dri_preds, all_dri_labels = [], []
                    valid_graphs = 0

                    for f in all_files:
                        data = torch.load(os.path.join(DATA_DIR, f))
                        split_path = os.path.join(SPLIT_DIR, f"multi_split_{f.replace('.pt', '')}.txt")
                        if not os.path.exists(split_path):
                            continue

                        with open(split_path, 'r') as file:
                            lines = file.read().splitlines()
                            train_line = lines[1].strip()
                            test_line = lines[3].strip()
                            train_indices = list(map(int, train_line.split(', ')))
                            test_indices = list(map(int, test_line.split(', ')))
                            train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
                            test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
                            train_mask[train_indices] = True
                            test_mask[test_indices] = True

                        data.train_mask = train_mask
                        data.test_mask = test_mask
                        data = data.to(device)

                        valid_train_mask = train_mask & ~torch.isnan(data.y[:, 0]) & ~torch.isnan(data.y[:, 1])
                        valid_test_mask = test_mask & ~torch.isnan(data.y[:, 0]) & ~torch.isnan(data.y[:, 1])

                        if valid_train_mask.sum() == 0 or valid_test_mask.sum() == 0:
                            continue

                        optimizer.zero_grad()
                        out = model(data)
                        loss = criterion(out[valid_train_mask], data.y[valid_train_mask])
                        loss.backward()
                        optimizer.step()

                        total_loss += loss.item()
                        valid_graphs += 1

                        with torch.no_grad():
                            all_fri_preds.append(out[valid_test_mask][:, 0].cpu())
                            all_dri_preds.append(out[valid_test_mask][:, 1].cpu())
                            all_fri_labels.append(data.y[valid_test_mask][:, 0].cpu())
                            all_dri_labels.append(data.y[valid_test_mask][:, 1].cpu())

                    if valid_graphs == 0:
                        print(f"⚠️ Epoch {epoch + 1}: No valid data.")
                        continue

                    avg_loss = total_loss / valid_graphs
                    train_losses.append(avg_loss)
                    writer.add_scalar("Loss/train", avg_loss, epoch + 1)

                    r2_fri = r2_score(torch.cat(all_fri_labels), torch.cat(all_fri_preds))
                    r2_dri = r2_score(torch.cat(all_dri_labels), torch.cat(all_dri_preds))
                    writer.add_scalars("R2", {'FRI': r2_fri, 'DRI': r2_dri}, epoch + 1)
                    csvwriter.writerow([epoch + 1, round(avg_loss, 6), round(r2_fri, 6), round(r2_dri, 6)])
                    print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, R² FRI={r2_fri:.4f}, R² DRI={r2_dri:.4f}")

                    # 保存两个任务的最优模型
                    if r2_fri > best_r2_fri:
                        best_r2_fri = r2_fri
                        for old in glob.glob(f'{SAVE_BEST_MODEL_DIR}/{model_type}_FRI_best_r2_*.pth'):
                            os.remove(old)
                        torch.save(model.state_dict(), f'{SAVE_BEST_MODEL_DIR}/{model_type}_FRI_best_r2_{r2_fri:.4f}.pth')

                    if r2_dri > best_r2_dri:
                        best_r2_dri = r2_dri
                        for old in glob.glob(f'{SAVE_BEST_MODEL_DIR}/{model_type}_DRI_best_r2_*.pth'):
                            os.remove(old)
                        torch.save(model.state_dict(), f'{SAVE_BEST_MODEL_DIR}/{model_type}_DRI_best_r2_{r2_dri:.4f}.pth')

                    early_stopper.step((r2_fri + r2_dri) / 2)
                    if early_stopper.early_stop:
                        print(f"⏹️ Early stopping at epoch {epoch + 1}")
                        break

            # 保存 Loss 曲线
            plt.figure()
            plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Training Loss - {LOG_NAME}')
            plt.grid()
            plt.savefig(CURVE_PNG)
            plt.close()

            print(f"📈 Loss curve saved to {CURVE_PNG}")
            writer.close()
