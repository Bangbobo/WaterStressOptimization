# ==== models.py ====
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, GINConv, TransformerConv
from torch_geometric.nn.aggr import MeanAggregation
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.models import MLP

# ======== GCN ========
class ThreeLayerGCNWithEdgeFeatures(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gcn3 = GCNConv(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.relu(self.gcn1(x, edge_index))
        x = self.dropout(x)
        x = self.relu(self.gcn2(x, edge_index))
        x = self.dropout(x)
        x = self.gcn3(x, edge_index)
        return x

# ======== GAT ========
class ThreeLayerGATWithEdgeFeatures(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, edge_feature_dim):
        super().__init__()
        self.edge_encoder = nn.Linear(edge_feature_dim, hidden_dim)
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=4, concat=True, dropout=0.3)
        self.gat2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True, dropout=0.3)
        self.gat3 = GATConv(hidden_dim * 4, output_dim, heads=1, concat=False, dropout=0.3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_feat = self.relu(self.edge_encoder(edge_attr))
        x = self.relu(self.proj(x))
        x = self.dropout(x)
        x = self.relu(self.gat1(x, edge_index, edge_attr=edge_feat))
        x = self.dropout(x)
        x = self.relu(self.gat2(x, edge_index, edge_attr=edge_feat))
        x = self.dropout(x)
        x = self.gat3(x, edge_index, edge_attr=edge_feat)
        return x

# ======== GIN ========
class ThreeLayerGIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        nn1 = MLP([input_dim, hidden_dim, hidden_dim])
        self.conv1 = GINConv(nn1)
        self.bn1 = BatchNorm(hidden_dim)

        nn2 = MLP([hidden_dim, hidden_dim, hidden_dim])
        self.conv2 = GINConv(nn2)
        self.bn2 = BatchNorm(hidden_dim)

        nn3 = MLP([hidden_dim, hidden_dim, output_dim])
        self.conv3 = GINConv(nn3)

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

# ======== Transformer ========
class ThreeLayerGraphTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, edge_feature_dim):
        super().__init__()
        self.edge_encoder = nn.Linear(edge_feature_dim, hidden_dim)
        self.node_encoder = nn.Linear(input_dim, hidden_dim)
        self.conv1 = TransformerConv(hidden_dim, hidden_dim, edge_dim=hidden_dim)
        self.conv2 = TransformerConv(hidden_dim, hidden_dim, edge_dim=hidden_dim)
        self.conv3 = TransformerConv(hidden_dim, output_dim, edge_dim=hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_feat = self.relu(self.edge_encoder(edge_attr))
        x = self.relu(self.node_encoder(x))
        x = self.dropout(x)
        x = self.relu(self.conv1(x, edge_index, edge_feat))
        x = self.dropout(x)
        x = self.relu(self.conv2(x, edge_index, edge_feat))
        x = self.dropout(x)
        x = self.conv3(x, edge_index, edge_feat)
        return x