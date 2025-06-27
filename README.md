# Ecohydrological Multi-Objective Optimization with Graph Neural Networks

This repository presents a robust, extensible framework for ecohydrological modeling using graph neural networks (GNNs), multi-task regression, and multi-objective optimization. Designed to address the challenge of balancing flood and drought mitigation in global basins, this project integrates domain-specific hydrological knowledge, GNN-based regression modeling, and evolutionary optimization using NSGA-II.

## 🔧 Features

| Feature                          | Status | Description                                                                 |
|----------------------------------|--------|-----------------------------------------------------------------------------|
| GCN/GAT/GIN/GraphTransformer     | ✅     | All implemented in `GNNs/models.py` with unified interfaces                 |
| EarlyStopping                    | ✅     | Automatic training termination based on validation R²                     |
| Validation R² Logging           | ✅     | R² is computed and saved at every epoch                                    |
| TensorBoard Support              | ✅     | Visualization logs saved to `results/tensorboard`                          |
| Label Distribution Analysis      | ✅     | Performed in `summary_report.ipynb`                                        |
| Multi-Task Regression            | ✅     | Optional (joint prediction of FRI and DRI)                                 |
| Basin-Level Context Injection    | ✅     | Mean elevation and other meta features included by default                 |
| Adversarial Masking              | ✅     | Train/test split via adversarial validation; masks saved to `.txt`         |
| Loss Curve Plotting              | ✅     | Auto-generated loss plot saved to `results/`                               |
| Model Comparison                 | ✅     | All four GNN variants trained and compared automatically                    |
| Configurable Parameters          | ✅     | Hidden dimension, epochs, learning rate are all configurable               |

---

## 📂 Project Structure

```bash
.
├── GNNs/
│   └── models.py              # GCN, GAT, GIN, GraphTransformer definitions
├── data_utils/
│   ├── multi_label_dataset.py # Generate multi-task dataset (FRI + DRI)
│   ├── adversarial_split.py   # Adversarial masking for mono-task
│   ├── split_loader.py        # Load saved train/test split masks
│   ├── split_masks/
│       └── multi/             # Stored .txt masks per basin
├── dataset_utils/
│   └── pyg_data_multi/        # PyG .pt files per basin (FRI + DRI)
├── tools/
│   ├── explainer_gcn.py       # Node & edge importance for GCN
│   ├── explainer_gat.py       # Node & edge importance for GAT
│   └── importance.py          # Convert .npy importance to .csv
├── train_model.py             # Unified training with config
├── ndmo.py                    # NSGA-II optimization using trained GNNs
├── summary_report.ipynb       # Data inspection and R² summary plots
├── requirements.txt           # All required packages
└── README.md
```

---

## 🧪 Dataset Preparation

```bash
# Step 1: Generate multi-label dataset
python data_utils/multi_label_dataset.py

# Step 2: Generate adversarial split masks for each basin
python data_utils/adversarial_split.py  # or adversarial_split_multi.py
```

---

## 🚀 Training

Train all models across all labels (FRI, DRI, or both):
```bash
python train_model.py
```

Results will be saved to `results/` including:
- Best model weights: `best_model_{MODEL}_{LABEL}.pth`
- Training logs: `train_log_{MODEL}_{LABEL}.csv`
- Loss curves: `loss_curve_{MODEL}_{LABEL}.png`
- TensorBoard logs: `tensorboard/{MODEL}_{LABEL}/`

### Training Configurations (in `train_model.py`):
```python
LABELS = ['FRI', 'DRI']           # or ['FRI'], or ['DRI'], or ['FRI', 'DRI']
MODEL_TYPES = ['GCN', 'GAT', 'GIN', 'GT']
MULTI_TASK = True                # Use FRI + DRI together
HIDDEN_DIM = 256                 # 128, 256, or 512
NUM_EPOCHS = 100
```

---

## 📈 Optimization (NSGA-II)

Once trained, you may run multi-objective optimization:
```bash
python ndmo.py
```
This will use the trained GNNs to compute FRI and DRI for multiple candidate flow solutions, then apply non-dominated sorting + crowding distance to extract Pareto fronts.

---

## 📊 Explainability

Run the GNNExplainer to obtain node and edge importance:
```bash
python tools/explainer_gat.py   # or explainer_gcn.py
```

To convert `.npy` importance maps to `.csv`:
```bash
python tools/importance.py
```

Visualizations are saved to `GNNExplainer/figs/`.

---

## 🧠 Research Highlights

- **Multi-task GNN Modeling:** Simultaneous prediction of flood and drought risk via shared spatial structure.
- **Adversarial Validation Splitting:** Ensures similarity between train/test distribution, avoiding data leakage.
- **Basin-Level Feature Injection:** Allows the model to access coarse-scale regional context, improving generalization.
- **Extensible Architecture:** Easily integrates other GNN variants and additional ecohydrological indices.

---

## 📚 Requirements

```bash
pip install -r requirements.txt
```

- torch
- torch_geometric
- numpy
- pandas
- scikit-learn
- matplotlib
- tensorboard

---

## 📬 Contact

- Author: 武丽蓉 (Tsinghua SIGS)
- Email: wlr22@mails.tsinghua.edu.cn

---

> This project is part of a Master's thesis on global ecohydrological optimization. Please cite appropriately if used in academic work.

