import torch
import torch.nn as nn
from .heads import Actor, Critic

class MultiCritics(nn.Module):
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
        # encoders 
        self.timeseries_encoder = timeseries_encoder    
        self.agent_encoder = agent_vector_encoder_cls(agent_input_dim, agent_hidden_dim, 
                                                      agent_out_dim, dropout)
        # fusion block 
        self.fusion_block = fusion_block_cls(embed_dim, agent_out_dim, 
                                             fusion_hidden_dim, dropout)
        # actor & critics 
        self.actor = Actor(fusion_hidden_dim+3, n_actions)
        self.critic_pnl     = Critic(fusion_hidden_dim)
        self.critic_risk    = Critic(fusion_hidden_dim)
        self.critic_regret  = Critic(fusion_hidden_dim)

    def forward(self, x, alpha):
        # data 
        ts_x, agent_x = x

        ts_out = self.timeseries_encoder(ts_x)
        agent_out = self.agent_encoder(agent_x)

        fusion_x = self.fusion_block((ts_out,agent_out))

        # critics: value 
        value_pnl    = self.critic_pnl(fusion_x)
        value_risk   = self.critic_risk(fusion_x)
        value_regret = self.critic_regret(fusion_x)

        # actor: policy logits 
        for_logits = torch.cat([fusion_x, alpha], dim=-1)
        logits = self.actor(for_logits)

        return logits, torch.cat([value_pnl, value_risk, value_regret], dim=-1)
