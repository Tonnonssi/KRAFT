from dataclasses import dataclass
from typing import ClassVar, List, Optional
import numpy as np

class DifferentialSharpeRatio:
    def __init__(self, span=300, initial_returns=None, epsilon=1e-8):
        if span < 1:
            raise ValueError("Span must be a positive integer.")
        
        self.eta = 2.0 / (span + 1)
        self.A = 0.0
        self.B = 1.0 # B의 초기값을 1.0으로 설정하여 초기 분산이 0이 되는 것을 방지
        self.epsilon = epsilon
        
        if initial_returns is not None and len(initial_returns) > 1:
            self.A = np.mean(initial_returns)
            self.B = np.mean(np.square(initial_returns))
            # 초기 분산이 너무 작으면 B를 약간 조정
            if (self.B - self.A**2) < self.epsilon:
                self.B = self.A**2 + self.epsilon
        
        print(f"DSR Initialized. A_0: {self.A:.6f}, B_0: {self.B:.6f}, eta: {self.eta:.6f}")

    def __call__(self, current_return):
        prev_A = self.A
        prev_B = self.B
        
        # 1. 분자 공식 수정: prev_B -> (prev_B - prev_A**2)
        variance = prev_B - prev_A**2
        
        # 2. 수치 안정성 강화
        if variance < self.epsilon:
            dsr = 0.0
        else:
            delta_A = current_return - prev_A
            delta_B = current_return**2 - prev_B
            numerator = variance * delta_A - 0.5 * prev_A * delta_B
            denominator = variance**(3/2)
            dsr = numerator / denominator
            
        self.A = prev_A + self.eta * (current_return - prev_A)
        self.B = prev_B + self.eta * (current_return**2 - prev_B)
        
        return dsr if not np.isnan(dsr) else 0.0
    

@dataclass
class RewardInfo:
    """보상 계산에 필요한 스텝별 데이터"""
    net_realized_pnl: float         # 손익 항목에 이용 
    prev_unrealized_pnl: float      # 손익 항목에 이용 
    current_unrealized_pnl: float   # 손익 항목에 이용 
    prev_balance: float             # 포트폴리오 가치에 이용(Risk)
    current_balance: float          # 포트폴리오 가치에 이용(Risk)
    prev_position: int              # 후회 항목에 이용 
    current_position: int           # 후회 항목에 이용 
    point_delta: float              # 후회 항목에 이용 
    execution_strength: int         # 만기일에 이용(Bonus)

    def __len__(self):
        return len(self.__dict__.values())


class RRPAReward:
    """RRPAReward (Risk Regret Profit Aware Reward)"""
    INIT_SEQ = [
        'w_profit', 'w_risk', 'w_regret',
        'margin_call_penalty', 'maturity_date_penalty',
        'bankrupt_penalty', 'goal_reward_bonus'
    ]

    def __init__(self, w_profit, w_risk, w_regret,
                 margin_call_penalty, maturity_date_penalty,
                 bankrupt_penalty, goal_reward_bonus):
        
        self.w_profit = w_profit
        self.w_risk = w_risk 
        self.w_regret = w_regret 
        self.margin_call_penalty = margin_call_penalty 
        self.maturity_date_penalty = maturity_date_penalty
        self.bankrupt_penalty = bankrupt_penalty 
        self.goal_reward_bonus = goal_reward_bonus 
        self.DSR = DifferentialSharpeRatio()
    
    @staticmethod
    def log(value):
        return np.sign(value) * np.log1p(abs(value) + 1e-6) 
    
    def _calculate_profit_reward(self, info: RewardInfo) -> float:
        """수익 컴포넌트(R_profit) 계산"""
        # 장기 
        realized_term = info.net_realized_pnl
        unrealized_term = info.current_unrealized_pnl - info.prev_unrealized_pnl
        # return self.log(realized_term) + self.log(unrealized_term)
        return self.log(realized_term + unrealized_term) # balance base

    def _calculate_risk_reward(self, info: RewardInfo) -> float:
        """위험 컴포넌트(R_risk) 계산"""
        portfolio_value_change = self.log(info.current_balance) - self.log(info.prev_balance)
        return self.DSR(portfolio_value_change)

    def _calculate_regret_penalty(self, info: RewardInfo) -> float:
        """후회(Regret) 페널티 계산"""
        is_flat = (np.sign(info.current_position) == 0 and np.sign(info.prev_position) == 0)
        return self.log(abs(info.point_delta)) if is_flat else 0.0

    def _apply_event_bonus_penalty(self, base_reward: float, event, reward_info) -> float:
        """이벤트에 따른 보너스 및 페널티 적용"""
        if 'margin_call' in event:
            return base_reward + self.margin_call_penalty
        if 'bankrupt' in event:
            return base_reward + self.bankrupt_penalty
        if 'goal_profit' in event:
            return base_reward + self.goal_reward_bonus
        if 'maturity_data' in event and reward_info.execution_strength != 0:
            return base_reward + self.maturity_date_penalty
        return base_reward
        
    def __call__(self, reward_info: RewardInfo, event: str) -> float:
        """
        주어진 데이터(info)와 이벤트(event)를 바탕으로 최종 보상을 계산.
        클래스 인스턴스를 함수처럼 호출할 수 있게 함.
        """
        r_profit = self._calculate_profit_reward(reward_info)
        r_risk = self._calculate_risk_reward(reward_info)
        r_regret = self._calculate_regret_penalty(reward_info)

        base_reward = (self.w_profit * r_profit +
                       self.w_risk * r_risk -
                       self.w_regret * r_regret)

        # NaN 또는 Inf 값이 나오면 0으로 처리
        if not np.isfinite(base_reward):
            base_reward = 0.0

        final_reward = self._apply_event_bonus_penalty(base_reward, event, reward_info)
        
        return np.clip(final_reward, -2,2)