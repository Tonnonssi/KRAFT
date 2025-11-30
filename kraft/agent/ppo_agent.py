import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np
from collections.abc import Mapping
from typing import Optional, Tuple


class _LinearEntropySchedule:
    """Simple helper for linear entropy-coefficient decay."""

    def __init__(self, start: float, end: float, steps: int):
        steps = int(steps)
        if steps <= 0:
            raise ValueError("Linear entropy schedule requires steps > 0")
        self.start = float(start)
        self.end = float(end)
        self.steps = steps

    def value_at(self, step: int) -> float:
        ratio = min(max(step, 0) / self.steps, 1.0)
        return self.start + (self.end - self.start) * ratio


class PPOAgent:
    INIT_SEQ = ['action_space', 'n_actions', 
                'value_coeff', 'entropy_coeff', 'clip_eps', 
                'gamma', 'lr', 'batch_size', 'train_epoch', 'device', 
                'entry_coeff', 'kappa', 'gae_lam',
                'beta', 'regulation']
    
    def __init__(self, model, action_space, n_actions, 
                value_coeff, entropy_coeff, clip_eps, 
                gamma, lr, batch_size, train_epoch, device, 
                entry_coeff, kappa, gae_lam,
                beta, regulation):
        '''
        PPOAgent 클래스 초기화 함수.
        모델, PPO 관련 계수들, 옵티마이저를 초기화한다.
        '''
        self.model = model.to(device)
        self.device = device

        # action params 
        self.action_space = action_space            # 행동공간 : [-k:k]
        self.n_actions = n_actions                  # 총행동수
        self.single_volume_cap = self.n_actions // 2# 동일포지션 최대계약수(k)

        # coeffs • epsilon 
        self.value_coeff = value_coeff              # 오류 함수 업데이트 시, 가치 반영 정도
        self._entropy_schedule = None
        self._entropy_schedule_step = 0
        self.entropy_coeff = self._init_entropy_coeff(entropy_coeff)
        self.clip_eps = clip_eps                    # 오류 함수 클리핑 정도 (PPO 핵심)

        # KL div related 
        self.entry_coeff = entry_coeff              # 오류 함수 업데이트 시, 진입 값 반영 정도 
        self.kappa = kappa                          # 트랜드 점수 민감도  
        self.beta = beta                            # 유니폼 분포(0.5)와의 혼합 비율
        self.regulation = regulation                # 규제의 정도  

        # discount params 
        self.gamma = gamma                          # 할인율 
        self.gae_lam = gae_lam                      # GAE를 단기적인 관점(0)에서 바라볼 건지, 장기적인 관점(1)에서 바라볼 건지 

        # train related 
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.critic_loss_ftn = nn.MSELoss()

        self.epoch = train_epoch                          # 에폭 크기 
        self.batch_size = batch_size                # 배치 크기 

    def _init_entropy_coeff(self, entropy_coeff):
        """Allow entropy coeff to be configured as scalar or schedule."""
        if isinstance(entropy_coeff, Mapping):
            return self._build_entropy_schedule(entropy_coeff)
        return float(entropy_coeff)

    def _build_entropy_schedule(self, spec: Mapping):
        if 'start' not in spec:
            raise ValueError("entropy_coeff.start is required when specifying a schedule")
        start = float(spec.get('start'))
        end = float(spec.get('end', start))
        raw_steps = spec.get('steps', spec.get('updates'))
        steps = int(raw_steps) if raw_steps is not None else 1
        steps = max(steps, 1)
        self._entropy_schedule = _LinearEntropySchedule(start, end, steps)
        return start

    def _entropy_coeff_for_update(self) -> float:
        if self._entropy_schedule is None:
            return self.entropy_coeff
        current = self._entropy_schedule.value_at(self._entropy_schedule_step)
        self.entropy_coeff = current
        return current

    def _advance_entropy_coeff(self):
        if self._entropy_schedule is None:
            return
        if self._entropy_schedule_step < self._entropy_schedule.steps:
            self._entropy_schedule_step += 1
        self.entropy_coeff = self._entropy_schedule.value_at(self._entropy_schedule_step)

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
        logits, _ = self.model(state)

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
        
    def clip_loss_ftn(self, 
                      advantage: torch.Tensor, 
                      old_log_prob: torch.Tensor, 
                      current_log_prob: torch.Tensor) -> torch.Tensor:
        '''
        ----------
        PPO의 clipped surrogate loss를 계산한다.

        - 현재 확률 대비 이전 확률의 비율을 계산하고,
          clip 범위 안에서 surrogate loss를 구한다.
        - 안정적인 policy 업데이트를 위함이다.
        '''
        ratio = torch.exp(current_log_prob - old_log_prob)
        ratio = torch.clamp(ratio, 1e-6, 1e2)           # 안정화 
        surrogate1 = ratio * advantage
        surrogate2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantage
        return torch.min(surrogate1, surrogate2).mean()
    
    def cal_advantage(self, memory, normalize=True):
        '''
        cal_advantage(memory: list[tuple], lam: float) -> torch.Tensor

        ----------
        Generalized Advantage Estimation(GAE)를 계산한다.

        - reversed list로 delta -> gae를 계산한다. 
        - GAE를 사용하면 bias-variance trade-off를 조절할 수 있다.
        '''

        rewards, dones, values, next_values = self._get_adantage_info(memory)
        values, next_values = values.squeeze(), next_values.squeeze()   # [Batch Size, 1] -> [Batch Size]

        # Generalize Advantage Estimate(GAE) calculation
        # reversed list로 delta -> gae를 계산. 
        advantage = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1.0 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lam * (1.0 - dones[t]) * gae
            advantage.insert(0, gae)

        adv = torch.tensor(advantage, dtype=torch.float32).unsqueeze(1)

        if normalize:
            mean, std = adv.mean(), adv.std()
            adv = (adv - mean) / (std + 1e-8)  # 분기 없이 안정적으로

        return adv
    
    def _get_adantage_info(self, memory):
        """advantage 계산에 필요한 정보들을 반환한다."""
        # 장기 
        # to.device() 뭐가 더 빠르냐 
        # set memory
        states, _, rewards, next_states, dones, _, _, _, _ = zip(*memory)

        # zip again to separate ts / agent
        ts_states, ag_states = zip(*states)
        n_ts_states, n_ag_states = zip(*next_states)

        # cat across batch dimension
        ts_states = torch.cat(ts_states, dim=0)
        ag_states = torch.cat(ag_states, dim=0)
        n_ts_states = torch.cat(n_ts_states, dim=0)
        n_ag_states = torch.cat(n_ag_states, dim=0)

        states, next_states = (ts_states, ag_states), (n_ts_states, n_ag_states)
        
        rewards = torch.cat(rewards, dim=0)
        dones = torch.cat(dones, dim=0)

        # get values - next_values : GAE 계산을 위함 
        with torch.no_grad():
            _, values = self.model(states)
            _, next_values = self.model(next_states)

        values = values.detach()
        next_values = next_values.detach()

        return rewards, dones, values, next_values
    
    def sample_memory(self, memory, advantage):
        """
        Memory와 advantage를 꺼내 trainable하게 만든다. 
        - tensor화, batch화, 학습할 Device에 올림 
        """
        states, decoded_actions, rewards, next_states, dones, old_log_probs, masks, entry_masks, entry_scores = zip(*memory)
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
        dones               = torch.cat(dones).to(self.device)
        old_log_probs       = torch.cat(old_log_probs).unsqueeze(1).to(self.device)
        masks               = torch.cat(masks).to(self.device)
        entry_masks         = torch.cat(entry_masks).to(self.device)
        entry_scores        = torch.cat(entry_scores).to(self.device)

        encoded_actions = self._encode_action(decoded_actions).to(self.device)
        return states, encoded_actions, rewards, next_states, dones, old_log_probs, advantages, masks, entry_masks, entry_scores

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
        current_entropy_coeff = self._entropy_coeff_for_update()

        for _ in range(self.epoch):
            # set memory
            states, encoded_actions, rewards, next_states, dones, old_log_probs, advantages, masks, entry_masks, entry_scores = self.sample_memory(memory, advantage)

            # get current values 
            self.model.train()
            current_logits, values = self.model(states)

            if masks is not None:
                # mask: shape [n_actions] with 1 (valid) or 0 (invalid)
                current_logits = current_logits.masked_fill(masks == 0, float('-inf'))

            # entropy bonus 
            action_dist = Categorical(logits=current_logits)                        
            current_log_probs = action_dist.log_prob(encoded_actions).unsqueeze(1)

            # KL Divergence 
            entry_reg = self._get_entry_regulation(current_logits, entry_masks, entry_scores)

            # 3 elements of loss : value_loss, clip_loss, entropy bonus 
            with torch.no_grad():
                _, next_values = self.model(next_states)
                value_target = rewards + self.gamma * next_values.squeeze() * (1 - dones)

            value_loss = self.critic_loss_ftn(values.squeeze(), value_target.detach())
            clip_loss = self.clip_loss_ftn(advantages.detach(), old_log_probs.detach(), current_log_probs)
            entropy = action_dist.entropy().mean()

            total_loss = -clip_loss + self.value_coeff * value_loss - current_entropy_coeff * entropy + self.entry_coeff * entry_reg

            losses += total_loss.item()

            # back-propagation 
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        self._advance_entropy_coeff()
        return losses / self.epoch
    
    def _get_entry_regulation(self, current_logits, entry_masks, entry_scores):
        '''현재 롱 숏 방향 분포에 대한 규제를 계산한다.'''
        # [1] 현재 롱 숏 방향 분포 
        current_policy_logit = current_logits.exp()
        entry_probs = current_policy_logit[entry_masks]

        short_probs = entry_probs[:, :self.single_volume_cap].sum(dim=1)
        long_probs = entry_probs[:, self.single_volume_cap:].sum(dim=1)

        total_probs = torch.stack([short_probs, long_probs], dim=1)
        sum_probs = total_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
        current_entry_policy = total_probs / sum_probs

        # [2] 상태별 트렌드 점수 s_t
        s = entry_scores[entry_masks]
        p_long = torch.sigmoid(self.kappa * s)
        p_target = torch.stack([1-p_long, p_long], dim=1)

        # [3] 극단치 방지를 위해 uniform prior와 섞음 
        p_mix = self.beta * 0.5 + (1-self.beta) * p_target

        # [4] trend가 확실하다면 규제를 약화 
        w = torch.sigmoid(-self.regulation * torch.abs(s)).detach()

        # [5] KL( 현재 정책 | 타깃 혼합 분포 )
        policy_safe = current_entry_policy.clamp_min(1e-6)
        p_mix_safe = p_mix.clamp_min(1e-6)
        kl = (policy_safe * (policy_safe.log() - p_mix_safe.log())).sum(dim=1)
        entry_reg = (w * kl).mean() 

        return entry_reg

    def load_model(self, state_dict):
        """외부에서 파라미터 업데이트를 도와주는 매서드"""
        self.model.load_state_dict(state_dict)

    def set_optimizer(self, new_optimizer):
        """기본으로 있는 옵티마이저를 변경할 때 이용하는 매서드"""
        self.optimizer = new_optimizer(self.model.parameters(), lr=self.lr)

    def _decode_action(self, encoded_action):
        """신경망 출력(0~20)을 에이전트 행동(-10~10)으로 변환."""
        return self.action_space[encoded_action]

    def _encode_action(self, decoded_action):
        """에이전트 행동(-10~10)을 신경망 출력(0~20)으로 변환."""
        offset = -self.action_space[0]  # -(-10) = 10 
        return decoded_action + offset
