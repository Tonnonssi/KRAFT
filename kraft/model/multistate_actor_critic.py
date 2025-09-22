import torch.nn as nn
from .heads import Actor, Critic

class MultiStatePV(nn.Module):
    INIT_SEQ = [
        'agent_input_dim', 'embed_dim', 'n_actions', 
        'agent_hidden_dim', 'agent_out_dim', 'fusion_hidden_dim', 'dropout'
    ]
    def __init__(self, 
                 timeseries_encoder: nn.Module,
                 agent_vector_encoder_cls,
                 fusion_block_cls,
                 agent_input_dim,        # agent 상태 feature 수
                 embed_dim,              # CNN + Transformer 임베딩 차원 (d_model)
                 n_actions, 
                 agent_hidden_dim=32, 
                 agent_out_dim=32,
                 fusion_hidden_dim=64,
                 dropout=0.1):             # dropout 비율        
        super().__init__()
        self.timeseries_encoder = timeseries_encoder
        
        self.agent_encoder = agent_vector_encoder_cls(agent_input_dim, agent_hidden_dim, 
                                                      agent_out_dim, dropout)

        self.fusion_block = fusion_block_cls(embed_dim, agent_out_dim, 
                                             fusion_hidden_dim, dropout)

        self.actor = Actor(fusion_hidden_dim, n_actions)
        self.critic = Critic(fusion_hidden_dim)

    def forward(self, x):
        ts_x, agent_x = x
        ts_out = self.timeseries_encoder(ts_x)
        agent_out = self.agent_encoder(agent_x)

        fusion_x = self.fusion_block((ts_out,agent_out))

        logits = self.actor(fusion_x)
        value = self.critic(fusion_x)
        
        return logits, value
