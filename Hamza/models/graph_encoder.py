import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool

class MoleculeGINE(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden=256, layers=5, out_dim=256):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden)
        self.edge_proj = nn.Linear(edge_dim, hidden)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINEConv(nn=mlp, edge_dim=hidden))
            self.norms.append(nn.LayerNorm(hidden))

        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_proj(x)
        e = self.edge_proj(edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, e)
            x = norm(x)
            x = F.relu(x)

        g = global_add_pool(x, batch)  # graph embedding
        return self.readout(g)
