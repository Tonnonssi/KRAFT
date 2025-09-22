import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    """
    가치를 구하는 함수
    """
    def __init__(self, fusion_hidden_dim):
        super().__init__()
        # critic params 
        self.critic_fc1 = nn.Linear(fusion_hidden_dim, fusion_hidden_dim)
        self.critic_fc2 = nn.Linear(fusion_hidden_dim, 1)

    def forward(self, x):
        critic_x = F.tanh(self.critic_fc1(x))
        value = self.critic_fc2(critic_x)
        return value 