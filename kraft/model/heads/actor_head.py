import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """
    정책을 출력하는 헤드
    """
    def __init__(self, fusion_hidden_dim, n_actions):
        super().__init__()
        # actor params 
        self.actor_fc1 = nn.Linear(fusion_hidden_dim, fusion_hidden_dim)
        self.actor_fc2 = nn.Linear(fusion_hidden_dim, n_actions)

    def forward(self, x):
        actor_x = F.tanh(self.actor_fc1(x))
        logits = self.actor_fc2(actor_x)
        return logits
