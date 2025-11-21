import torch 
import torch.nn as nn
from torch.distributions.categorical import Categorical
from typing import Optional
from .ppo_agent import PPOAgent 

class MultiCriticsPPOAgent(PPOAgent):
    """single과 거의 유사하지만, model의 input 값이 달라서 대부분 재정의함"""

    def get_alpha(self, alpha):
        """현재 에피소드에서 사용하는 alpha 값을 설정"""
        if isinstance(alpha, torch.Tensor):
            alpha_tensor = alpha.clone().detach().to(torch.float32)
        else:
            alpha_tensor = torch.as_tensor(alpha, dtype=torch.float32)
        self.alpha = alpha_tensor.to(self.device).unsqueeze(0) # [1, 3]

    def get_action(self, 
                   state:torch.Tensor, 
                   mask:torch.Tensor=None, 
                   stochastic:bool=True) -> tuple[int, Optional[float]]:
        '''
        ----------
        - mask: shape [n_actions] with 1 (valid) or 0 (invalid)
                dtype bool
        - stochastic opt : available determinstic choice 
        - agent가 사용하는 action 체계를 이용하기 위해, action decode를 진행 
        '''
        def _to_device_tensor(arr):
            if isinstance(arr, torch.Tensor):
                return arr.to(self.device)
            tensor = torch.as_tensor(arr, dtype=torch.float32)
            return tensor.to(self.device)

        state = tuple(_to_device_tensor(s) for s in state)
        logits, _ = self.model(state, self.alpha)

        if mask is not None:
            # 이거 장기적으로 수정 
            mask = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(self.device)
            logits = logits.masked_fill(mask == 0, float('-inf'))
        
        if stochastic:
            action_dist = Categorical(logits=logits)           # for entropy bonus
            encoded_action = action_dist.sample()              # 정책에 따라 행동 추출
            log_prob = action_dist.log_prob(encoded_action)    # 행동의 logit 값

            decoded_action = self._decode_action(encoded_action.item())
            return decoded_action, log_prob.item()
        else:
            # determinstic opt. 
            encoded_action = torch.argmax(logits, dim=-1)
            decoded_action = self._decode_action(encoded_action.item())
            return decoded_action, None

    def _get_aggregated_advantages(self, advantages, alpha):
        """
        advantages: [T, 3]
        alpha:      [T, 3]   
        return:     [T, 1]
        """
        weighted_advantages = advantages * alpha
        return weighted_advantages.sum(dim=-1, keepdim=True)  # [T, 1]

    def cal_advantage(self, memory, normalize=True):
        '''변경한 부분: A에서 사용하는 가치를 alpha에 따라 가중합하여 scalar로 변환'''
        rewards, dones, values, next_values, alpha = self._get_adantage_info(memory)

        advantage = torch.zeros_like(rewards, dtype=torch.float32).to(self.device)  # [Batch, 3]
        gae = torch.zeros((rewards.size(1),), dtype=torch.float32).to(self.device)  # [3,]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1.0 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lam * (1.0 - dones[t]) * gae
            advantage[t] = gae

        adv = self._get_aggregated_advantages(advantage, alpha)

        if normalize:
            mean, std = adv.mean(), adv.std()
            adv = (adv - mean) / (std + 1e-8)  # 분기 없이 안정적으로
        return adv
        
    def train(self, 
              memory: list[tuple], 
              advantage: torch.Tensor) -> float:
        '''
        train(memory: list[tuple], advantage: torch.Tensor) -> float

        ----------
        PPO 손실 함수를 계산하고 모델 파라미터를 업데이트한다.

        - PPO 기본 오류 함수에 KL div를 추가했다: 
          (1) value loss, (2) clipped surrogate loss, (3) entropy bonus (4) KL div for entry explore
        - GAE로 계산된 advantage를 기반으로 policy와 value를 모두 학습한다.
        '''
        if len(memory) < self.batch_size:
            return 0
        
        losses = 0

        for _ in range(self.epoch):
            # set memory
            # ===============================================================
            # single과 달리 alpha 항이 더 있음 
            states, encoded_actions, rewards, next_states, dones, old_log_probs, advantages, masks, entry_masks, entry_scores, alpha = self.sample_memory(memory, advantage)

            # get current values 
            self.model.train()
            current_logits, values = self.model(states, alpha)

            if masks is not None:
                # mask: shape [n_actions] with 1 (valid) or 0 (invalid)
                current_logits = current_logits.masked_fill(masks == 0, float('-inf'))

            # entropy bonus 
            action_dist = Categorical(logits=current_logits)                        
            current_log_probs = action_dist.log_prob(encoded_actions).unsqueeze(1)

            # KL Divergence based regularization
            # entry_reg = self._get_entry_regulation(current_logits, entry_masks, entry_scores)

            # 3 elements of loss : value_loss, clip_loss, entropy bonus 
            with torch.no_grad():
                _, next_values = self.model(next_states, alpha)
                value_target = rewards + self.gamma * next_values * (1 - dones)

            value_loss = self.critic_loss_ftn(values, value_target.detach())
            clip_loss = self.clip_loss_ftn(advantages.detach(), old_log_probs.detach(), current_log_probs)
            entropy = action_dist.entropy().mean()

            total_loss = -clip_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy # + self.entry_coeff * entry_reg

            losses += total_loss.item()

            # back-propagation 
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        return losses / self.epoch
    
    def _get_adantage_info(self, memory):
        """advantage 계산에 필요한 정보들을 반환한다."""
        # 장기 
        # to.device() 뭐가 더 빠르냐 
        # set memory
        # ===============================================================
        # single과 달리 alpha 항이 더 있음 
        states, _, rewards, next_states, dones, _, _, _, _, alpha = zip(*memory)

        # zip again to separate ts / agent
        ts_states, ag_states        = zip(*states)
        n_ts_states, n_ag_states    = zip(*next_states)

        # cat across batch dimension
        ts_states   = torch.cat(ts_states, dim=0)
        ag_states   = torch.cat(ag_states, dim=0)
        n_ts_states = torch.cat(n_ts_states, dim=0)
        n_ag_states = torch.cat(n_ag_states, dim=0)

        states, next_states = (ts_states, ag_states), (n_ts_states, n_ag_states)
        
        rewards = torch.cat(rewards, dim=0).to(self.device)
        dones   = torch.cat(dones, dim=0).unsqueeze(-1).to(self.device)
        alpha   = torch.cat(alpha, dim=0).to(self.device)

        # get values - next_values : GAE 계산을 위함 
        with torch.no_grad():
            _, values       = self.model(states, alpha)
            _, next_values  = self.model(next_states, alpha)

        values, next_values = values.detach(), next_values.detach()

        return rewards, dones, values, next_values, alpha
    
    def sample_memory(self, memory, advantage):
        """
        Memory와 advantage를 꺼내 trainable하게 만든다. 
        - tensor화, batch화, 학습할 Device에 올림 
        """
        states, decoded_actions, rewards, next_states, dones, old_log_probs, masks, entry_masks, entry_scores, alpha = zip(*memory)
        advantages = advantage.to(self.device)

        # zip again to separate ts / agent
        ts_states, ag_states = zip(*states)
        n_ts_states, n_ag_states = zip(*next_states)

        # cat across batch dimension
        ts_states           = torch.cat(ts_states, dim=0).to(self.device)
        ag_states           = torch.cat(ag_states, dim=0).to(self.device)
        n_ts_states         = torch.cat(n_ts_states, dim=0).to(self.device)
        n_ag_states         = torch.cat(n_ag_states, dim=0).to(self.device)

        states, next_states = (ts_states, ag_states), (n_ts_states, n_ag_states)

        decoded_actions     = torch.cat(decoded_actions)
        rewards             = torch.cat(rewards, dim=0).to(self.device)
        dones               = torch.cat(dones, dim=0).to(self.device)
        if dones.dim() == 1:
            dones = dones.unsqueeze(-1)
        old_log_probs       = torch.cat(old_log_probs).unsqueeze(1).to(self.device)
        masks               = torch.cat(masks).to(self.device)
        entry_masks         = torch.cat(entry_masks).to(self.device)
        entry_scores        = torch.cat(entry_scores).to(self.device)

        alpha               = torch.cat(alpha).to(self.device)    

        encoded_actions = self._encode_action(decoded_actions).to(self.device)
        return states, encoded_actions, rewards, next_states, dones, old_log_probs, advantages, masks, entry_masks, entry_scores, alpha
