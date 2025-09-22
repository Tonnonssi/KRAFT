import torch
import torch.nn as nn

class BaseFusion(nn.Module):
    def __init__(self,
                 timeseries_embed_dim=64,
                 agent_out_dim=32,
                 fusion_hidden_dim=64,
                 dropout=0.1):
        super().__init__()

        # Fusion MLP layers
        self.fusion_fc1 = nn.Linear(timeseries_embed_dim + agent_out_dim, fusion_hidden_dim)
        self.fusion_relu = nn.ReLU()
        self.fusion_dropout = nn.Dropout(dropout)
        self.fusion_fc2 = nn.Linear(fusion_hidden_dim, fusion_hidden_dim)

    def forward(self, x):
        ts_out, agent_out = x
        
        fused = torch.cat([ts_out, agent_out], dim=1)  # (B, embed + agent_out)

        x = self.fusion_fc1(fused)
        x = self.fusion_relu(x)
        x = self.fusion_dropout(x)
        x = self.fusion_fc2(x)
        return x